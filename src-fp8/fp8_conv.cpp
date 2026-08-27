#include "fp8_conv.hpp"
#include "darknet_internal.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace Darknet
{
	int fp8_conv_out_dim(const int input, const int pad, const int dilation, const int kernel, const int stride)
	{
		return (input + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;
	}
}

#if defined(DARKNET_GPU_CUDA) && defined(DARKNET_FP8_CUDNN_CONV) && defined(CUDART_VERSION) && CUDART_VERSION >= 12010

#include <cudnn_frontend.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace fe = cudnn_frontend;

namespace Darknet
{
	struct Fp8ConvPlan
	{
		Fp8ConvSpec spec;
		int out_h = 0;
		int out_w = 0;
		size_t workspace_bytes = 0;
		std::shared_ptr<fe::graph::Graph> graph;
		std::shared_ptr<fe::graph::Tensor_attributes> x;
		std::shared_ptr<fe::graph::Tensor_attributes> w;
		std::shared_ptr<fe::graph::Tensor_attributes> y;
		std::shared_ptr<fe::graph::Tensor_attributes> input_scale;
		std::shared_ptr<fe::graph::Tensor_attributes> weight_scale;
		std::shared_ptr<fe::graph::Tensor_attributes> bias;
		const float * input_scale_gpu = nullptr;
		const float * weight_scale_gpu = nullptr;
	};

	namespace
	{
		std::vector<int64_t> nhwc_stride(const int64_t n, const int64_t c, const int64_t h, const int64_t w)
		{
			(void)n;
			return {h * w * c, 1, w * c, c};
		}

		fe::DataType_t output_type(const Fp8ConvOutput output)
		{
			switch (output)
			{
				case Fp8ConvOutput::Fp32: return fe::DataType_t::FLOAT;
				case Fp8ConvOutput::Bf16: return fe::DataType_t::BFLOAT16;
			}
			return fe::DataType_t::FLOAT;
		}

		std::shared_ptr<fe::graph::Tensor_attributes> scalar_tensor(std::shared_ptr<fe::graph::Graph> const & graph, const char * name)
		{
			return graph->tensor(fe::graph::Tensor_attributes()
				.set_name(name)
				.set_dim({1, 1, 1, 1})
				.set_stride({1, 1, 1, 1})
				.set_data_type(fe::DataType_t::FLOAT));
		}

		bool debug_enabled()
		{
			return std::getenv("DARKNET_FP8_DEBUG") != nullptr;
		}

		bool log_frontend_failure(const char * stage, fe::error_t status, const Fp8ConvSpec & spec)
		{
			if (status.is_good())
			{
				return true;
			}
			if (debug_enabled())
			{
				std::fprintf(
					stderr,
					"fp8_conv %s failed: %s shape=n%d c%d h%d w%d k%d r%d s%d stride=%dx%d pad=%dx%d dilation=%dx%d output=%d bias=%d relu=%d\n",
					stage,
					status.get_message().c_str(),
					spec.batch,
					spec.channels,
					spec.height,
					spec.width,
					spec.filters,
					spec.kernel_h,
					spec.kernel_w,
					spec.stride_h,
					spec.stride_w,
					spec.pad_h,
					spec.pad_w,
					spec.dilation_h,
					spec.dilation_w,
					static_cast<int>(spec.output),
					spec.fuse_bias ? 1 : 0,
					spec.fuse_relu ? 1 : 0);
			}
			return false;
		}

		bool finalize_plan(Fp8ConvPlan * plan)
		{
			auto & graph = *plan->graph;
			if (!log_frontend_failure("validate", graph.validate(), plan->spec))
			{
				return false;
			}
			if (!log_frontend_failure("build_operation_graph", graph.build_operation_graph(cudnn_handle()), plan->spec))
			{
				return false;
			}
			if (!log_frontend_failure("heuristics_a", graph.create_execution_plans({fe::HeurMode_t::A}), plan->spec))
			{
				return false;
			}
			if (!log_frontend_failure("check_support", graph.check_support(cudnn_handle()), plan->spec))
			{
				return false;
			}
			if (!log_frontend_failure("build_plan", graph.build_plans(cudnn_handle(), fe::BuildPlanPolicy_t::HEURISTICS_CHOICE), plan->spec))
			{
				return false;
			}

			int64_t workspace = 0;
			if (!log_frontend_failure("workspace", graph.get_workspace_size(workspace), plan->spec))
			{
				return false;
			}
			plan->workspace_bytes = static_cast<size_t>(std::max<int64_t>(workspace, 0));

			if (debug_enabled())
			{
				std::string plan_name;
				if (graph.get_plan_name(plan_name).is_good())
				{
					std::fprintf(stderr, "fp8_conv fprop selected %s workspace=%zu\n", plan_name.c_str(), plan->workspace_bytes);
				}
			}
			return true;
		}
	}

	bool fp8_conv_supported()
	{
		if (std::getenv("DARKNET_FP8_DISABLE_CUDNN_CONV") != nullptr)
		{
			return false;
		}

		int device = 0;
		cudaError_t status = cudaGetDevice(&device);
		if (status != cudaSuccess)
		{
			return false;
		}

		cudaDeviceProp prop = {};
		status = cudaGetDeviceProperties(&prop, device);
		if (status != cudaSuccess)
		{
			return false;
		}

		const int capability = prop.major * 100 + prop.minor * 10;
		return capability >= 890 && cudnnGetVersion() >= 91700;
	}

	Fp8ConvPlan * fp8_conv_plan_create_fprop(
		const Fp8ConvSpec & spec,
		const float * input_scale_gpu,
		const float * weight_scale_gpu)
	{
		if (!fp8_conv_supported() ||
			spec.batch <= 0 ||
			spec.channels <= 0 ||
			spec.height <= 0 ||
			spec.width <= 0 ||
			spec.filters <= 0 ||
			spec.kernel_h <= 0 ||
			spec.kernel_w <= 0 ||
			spec.stride_h <= 0 ||
			spec.stride_w <= 0 ||
			spec.dilation_h <= 0 ||
			spec.dilation_w <= 0 ||
			input_scale_gpu == nullptr ||
			weight_scale_gpu == nullptr)
		{
			return nullptr;
		}

		const int out_h = fp8_conv_out_dim(spec.height, spec.pad_h, spec.dilation_h, spec.kernel_h, spec.stride_h);
		const int out_w = fp8_conv_out_dim(spec.width, spec.pad_w, spec.dilation_w, spec.kernel_w, spec.stride_w);
		if (out_h <= 0 || out_w <= 0)
		{
			return nullptr;
		}

		auto * plan = new Fp8ConvPlan;
		plan->spec = spec;
		plan->out_h = out_h;
		plan->out_w = out_w;
		plan->input_scale_gpu = input_scale_gpu;
		plan->weight_scale_gpu = weight_scale_gpu;
		plan->graph = std::make_shared<fe::graph::Graph>();
		plan->graph->set_io_data_type(fe::DataType_t::FLOAT)
			.set_intermediate_data_type(fe::DataType_t::FLOAT)
			.set_compute_data_type(fe::DataType_t::FLOAT);

		plan->x = plan->graph->tensor(fe::graph::Tensor_attributes()
			.set_name("x")
			.set_dim({spec.batch, spec.channels, spec.height, spec.width})
			.set_stride(nhwc_stride(spec.batch, spec.channels, spec.height, spec.width))
			.set_data_type(fe::DataType_t::FP8_E4M3));
		plan->w = plan->graph->tensor(fe::graph::Tensor_attributes()
			.set_name("w")
			.set_dim({spec.filters, spec.channels, spec.kernel_h, spec.kernel_w})
			.set_stride({spec.channels * spec.kernel_h * spec.kernel_w, 1, spec.kernel_w * spec.channels, spec.channels})
			.set_data_type(fe::DataType_t::FP8_E4M3));

		auto conv = plan->graph->conv_fprop(
			plan->x,
			plan->w,
			fe::graph::Conv_fprop_attributes()
				.set_padding({spec.pad_h, spec.pad_w})
				.set_stride({spec.stride_h, spec.stride_w})
				.set_dilation({spec.dilation_h, spec.dilation_w})
				.set_name("conv"));

		plan->input_scale = scalar_tensor(plan->graph, "input_scale");
		plan->weight_scale = scalar_tensor(plan->graph, "weight_scale");
		auto out = plan->graph->pointwise(conv, plan->input_scale, fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::MUL));
		out = plan->graph->pointwise(out, plan->weight_scale, fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::MUL));

		if (spec.fuse_bias)
		{
			plan->bias = plan->graph->tensor(fe::graph::Tensor_attributes()
				.set_name("bias")
				.set_dim({1, spec.filters, 1, 1})
				.set_stride({spec.filters, 1, spec.filters, spec.filters})
				.set_data_type(fe::DataType_t::FLOAT));
			out = plan->graph->pointwise(out, plan->bias, fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD));
		}
		if (spec.fuse_relu)
		{
			out = plan->graph->pointwise(out, fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::RELU_FWD));
		}
		plan->y = out;
		plan->y->set_output(true)
			.set_data_type(output_type(spec.output))
			.set_dim({spec.batch, spec.filters, out_h, out_w})
			.set_stride(nhwc_stride(spec.batch, spec.filters, out_h, out_w));

		if (!finalize_plan(plan))
		{
			fp8_conv_plan_destroy(plan);
			return nullptr;
		}
		return plan;
	}

	void fp8_conv_plan_destroy(Fp8ConvPlan * plan)
	{
		delete plan;
	}

	size_t fp8_conv_workspace_bytes(const Fp8ConvPlan * plan)
	{
		return plan ? plan->workspace_bytes : 0;
	}

	bool fp8_conv_output_is_bf16(const Fp8ConvPlan * plan)
	{
		return plan && plan->spec.output == Fp8ConvOutput::Bf16;
	}

	bool fp8_conv_fuses_bias(const Fp8ConvPlan * plan)
	{
		return plan && plan->spec.fuse_bias;
	}

	bool fp8_conv_fuses_relu(const Fp8ConvPlan * plan)
	{
		return plan && plan->spec.fuse_relu;
	}

	bool fp8_conv_fprop(
		Fp8ConvPlan * plan,
		const void * input_fp8_nhwc_gpu,
		const void * weights_fp8_krsc_gpu,
		const float * bias_gpu,
		void * output_nhwc_gpu,
		void * workspace,
		const size_t workspace_bytes)
	{
		TAT(TATPARMS);

		if (plan == nullptr ||
			input_fp8_nhwc_gpu == nullptr ||
			weights_fp8_krsc_gpu == nullptr ||
			output_nhwc_gpu == nullptr ||
			plan->workspace_bytes > workspace_bytes ||
			(plan->workspace_bytes > 0 && workspace == nullptr) ||
			(plan->bias != nullptr && bias_gpu == nullptr))
		{
			return false;
		}

		std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void *> variant_pack = {
			{plan->x, const_cast<void *>(input_fp8_nhwc_gpu)},
			{plan->w, const_cast<void *>(weights_fp8_krsc_gpu)},
			{plan->y, output_nhwc_gpu},
			{plan->input_scale, const_cast<float *>(plan->input_scale_gpu)},
			{plan->weight_scale, const_cast<float *>(plan->weight_scale_gpu)}
		};
		if (plan->bias)
		{
			variant_pack.emplace(plan->bias, const_cast<float *>(bias_gpu));
		}

		auto status = plan->graph->execute(cudnn_handle(), variant_pack, workspace);
		if (status.is_bad())
		{
			log_frontend_failure("execute", status, plan->spec);
			return false;
		}
		return true;
	}
}

#else

namespace Darknet
{
	bool fp8_conv_supported()
	{
		return false;
	}

	Fp8ConvPlan * fp8_conv_plan_create_fprop(const Fp8ConvSpec &, const float *, const float *)
	{
		return nullptr;
	}

	void fp8_conv_plan_destroy(Fp8ConvPlan *) {}

	size_t fp8_conv_workspace_bytes(const Fp8ConvPlan *)
	{
		return 0;
	}

	bool fp8_conv_output_is_bf16(const Fp8ConvPlan *)
	{
		return false;
	}

	bool fp8_conv_fuses_bias(const Fp8ConvPlan *)
	{
		return false;
	}

	bool fp8_conv_fuses_relu(const Fp8ConvPlan *)
	{
		return false;
	}

	bool fp8_conv_fprop(Fp8ConvPlan *, const void *, const void *, const float *, void *, void *, size_t)
	{
		return false;
	}
}

#endif
