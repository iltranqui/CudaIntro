/* Darknet/YOLO:  https://codeberg.org/CCodeRun/darknet
 * Copyright 2024-2026 Stephane Charette
 */

#include "darknet_internal.hpp"
#include "convolutional_layer.hpp"
#ifdef DARKNET_HAS_FP4
#include "fp4_gemm.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>


namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();


	std::string requested_precision()
	{
		if (cfg_and_state.is_set("fp4"))
		{
			return "fp4";
		}
		if (cfg_and_state.is_set("fp8"))
		{
			return "fp8";
		}
		if (cfg_and_state.is_set("bf16"))
		{
			return "bf16";
		}

		return "config-default";
	}


	std::string effective_precision(const Darknet::Network & net)
	{
		if (net.fp4_inference)
		{
			return "fp4";
		}
		if (net.fp8_inference)
		{
			return "fp8";
		}
		if (net.cudnn_bf16)
		{
			return "bf16";
		}
		if (net.cudnn_half)
		{
			return "fp16";
		}

		return "fp32";
	}


	std::vector<float> make_synthetic_input(const size_t count)
	{
		std::vector<float> input(count);
		uint32_t state = 0x4d595df4u;
		for (size_t idx = 0; idx < count; ++idx)
		{
			// Fixed, non-zero values make this independent of image I/O while still
			// exercising activation quantization and tensor-core paths.
			state = state * 1664525u + 1013904223u;
			input[idx] = static_cast<float>((state >> 8) & 0x00ffffffu) / 8388607.5f - 1.0f;
		}

		return input;
	}


	bool enable_missing_fp8_scales(Darknet::Network & net)
	{
		// FP8 inference normally obtains activation scales from a sidecar loaded
		// with real weights.  For a benchmark with no weights, use a unit input
		// scale so the code path can still be timed.  Never overwrite calibrated
		// scales that load_weights() has already supplied.
		bool used_synthetic_scales = false;
		for (int idx = 0; idx < net.n; ++idx)
		{
			Darknet::Layer & layer = net.layers[idx];
			if (layer.type == Darknet::ELayerType::CONVOLUTIONAL && !layer.fp8_scales_loaded)
			{
				layer.fp8_scales_loaded = 1;
				layer.fp8_input_scale_host = 1.0f;
				used_synthetic_scales = true;
			}
		}
		return used_synthetic_scales;
	}


#ifdef DARKNET_HAS_FP4
	struct Fp4DispatchCounts
	{
		std::uint64_t cudnn_frontend = 0;
		std::uint64_t cublaslt = 0;
		std::uint64_t relay = 0;
		std::uint64_t failed = 0;
	};


	Fp4DispatchCounts fp4_dispatch_counts(const Darknet::Network & net)
	{
		Fp4DispatchCounts totals;
		for (int idx = 0; idx < net.n; ++idx)
		{
			const Darknet::Layer & layer = net.layers[idx];
			if (layer.fp4_gemm_plan)
			{
				const auto report = Darknet::fp4_gemm_report(
					static_cast<const Darknet::Fp4GemmPlan *>(layer.fp4_gemm_plan));
				totals.cudnn_frontend += report.cudnn_frontend_executions;
				totals.cublaslt += report.cublaslt_executions;
				totals.relay += report.prequantized_right_executions;
				totals.failed += report.failed_executions;
			}
		}
		return totals;
	}


	Fp4DispatchCounts operator-(const Fp4DispatchCounts & lhs, const Fp4DispatchCounts & rhs)
	{
		return {lhs.cudnn_frontend - rhs.cudnn_frontend,
			lhs.cublaslt - rhs.cublaslt,
			lhs.relay - rhs.relay,
			lhs.failed - rhs.failed};
	}
#endif


	void force_single_stream(Darknet::Network & net)
	{
		// A CUDA event on net's stream measures all work only when the cfg has
		// not moved individual layers onto legacy switch streams.  The benchmark
		// therefore deliberately uses one stream for repeatable whole-network
		// comparisons.
		for (int idx = 0; idx < net.n; ++idx)
		{
			net.layers[idx].stream = -1;
			net.layers[idx].wait_stream_id = -1;
		}
	}


	struct TimingSummary
	{
		float mean_ms = 0.0f;
		float median_ms = 0.0f;
		float p95_ms = 0.0f;
	};


	TimingSummary summarize_timings(std::vector<float> timings)
	{
		TimingSummary summary;
		if (timings.empty())
		{
			return summary;
		}

		float sum = 0.0f;
		for (const float value : timings)
		{
			sum += value;
		}
		summary.mean_ms = sum / static_cast<float>(timings.size());

		std::sort(timings.begin(), timings.end());
		const size_t median_index = timings.size() / 2;
		summary.median_ms = timings[median_index];
		const size_t p95_index = static_cast<size_t>(std::ceil(timings.size() * 0.95f)) - 1U;
		summary.p95_ms = timings[std::min(p95_index, timings.size() - 1U)];

		return summary;
	}
}


void run_precision_benchmark()
{
#ifndef DARKNET_GPU_CUDA
	darknet_fatal_error(DARKNET_LOC, "precision-benchmark requires a CUDA build");
#else
	if (cfg_and_state.cfg_filename.empty())
	{
		darknet_fatal_error(DARKNET_LOC, "precision-benchmark requires a .cfg file");
	}

	const int batch = cfg_and_state.get("precisionbatch", 4);
	const int warmup = cfg_and_state.get("precisionwarmup", 30);
	const int iterations = cfg_and_state.get("precisioniterations", 200);
	if (batch <= 0 || warmup < 0 || iterations <= 0)
	{
		darknet_fatal_error(DARKNET_LOC,
			"precision-benchmark requires batch > 0, warmup >= 0, and iterations > 0");
	}

	Darknet::Network net = parse_network_cfg_custom(cfg_and_state.cfg_filename.string().c_str(), batch, 1);
	if (net.gpu_index < 0)
	{
		free_network(net);
		darknet_fatal_error(DARKNET_LOC, "precision-benchmark requires an active CUDA GPU");
	}

	// Do not benchmark CUDA graph capture/replay.  This is a direct, device-only
	// forward pass and is comparable across BF16, FP8, and FP4 configurations.
	net.use_cuda_graph = 0;
	force_single_stream(net);
	activate_network_streams(net);

	const bool weights_loaded = !cfg_and_state.weights_filename.empty();
	if (weights_loaded)
	{
		load_weights(&net, cfg_and_state.weights_filename.string().c_str());
	}

	// Match the normal inference preparation order.  If no weights are passed,
	// Darknet's initialized weights are measured and the output marks that fact.
	fuse_conv_batchnorm(net);
	// Synthetic scales are useful only for an initialized-weights smoke
	// benchmark.  With a real weights file, preserve the production rule that
	// FP8 inference requires its calibration sidecar.
	const bool synthetic_fp8_scales = !weights_loaded && enable_missing_fp8_scales(net);
	calculate_binary_weights(&net);

	const std::string requested = requested_precision();
	const std::string effective = effective_precision(net);
	if (requested == "fp4" && !net.fp4_inference)
	{
		deactivate_network_streams();
		free_network(net);
		darknet_fatal_error(DARKNET_LOC, "FP4 was requested but no FP4 inference path was enabled");
	}
	if (requested == "fp8" && !net.fp8_inference)
	{
		deactivate_network_streams();
		free_network(net);
		darknet_fatal_error(DARKNET_LOC, "FP8 was requested but no FP8 inference path was enabled");
	}
	if (requested == "bf16" && !net.cudnn_bf16)
	{
		deactivate_network_streams();
		free_network(net);
		darknet_fatal_error(DARKNET_LOC, "BF16 was requested but cuDNN BF16 was not enabled");
	}

	int fp4_eligible = 0;
	int fp4_prepacked = 0;
	int fp4_relay_links = 0;
	int fp8_eligible = 0;
	int fp8_prepacked = 0;
	int fp8_relay_links = 0;
	int fp8_relay_enabled_links = 0;
	for (int idx = 0; idx < net.n; ++idx)
	{
		const Darknet::Layer & layer = net.layers[idx];
		fp4_eligible += layer.fp4_eligible != 0;
		fp4_prepacked += layer.fp4_weights_prepacked != 0;
#ifdef DARKNET_HAS_FP4
		fp4_relay_links += net.fp4_inference &&
			layer.type == Darknet::ELayerType::CONVOLUTIONAL &&
			layer.fp4_relay_next_layer >= 0;
#endif
		fp8_eligible += layer.fp8_eligible != 0;
		fp8_prepacked += layer.weights_fp8_gpu != nullptr || layer.weights_fp8_nhwc_gpu != nullptr;
		fp8_relay_links += net.fp8_inference &&
			layer.type == Darknet::ELayerType::CONVOLUTIONAL &&
			layer.fp8_relay_next_layer >= 0;
	}

	const size_t input_count = static_cast<size_t>(get_network_input_size(net)) * net.batch;
	std::vector<float> synthetic_input = make_synthetic_input(input_count);
	float * input_gpu = cuda_make_array(synthetic_input.data(), input_count);
	const cudaStream_t stream = get_network_cuda_stream(net);
	CHECK_CUDA(cudaStreamSynchronize(stream));

	auto enqueue_forward = [&]()
	{
		Darknet::NetworkState state = {0};
		state.index = 0;
		state.net = net;
		state.input = input_gpu;
		state.train = 0;
		forward_network_gpu(net, state);
	};

	for (int idx = 0; idx < warmup; ++idx)
	{
		enqueue_forward();
		CHECK_CUDA(cudaStreamSynchronize(stream));
#ifdef DARKNET_HAS_FP8
		fp8_finalize_network_activation_calibration(net);
#endif
		reset_wait_stream_events();
	}
	if (net.fp8_activation_calibration_pending)
	{
		// `precisionwarmup=0` remains valid, but the precision plan still needs
		// one untimed eager frame before the measured graph can use a relay.
		enqueue_forward();
		CHECK_CUDA(cudaStreamSynchronize(stream));
#ifdef DARKNET_HAS_FP8
		fp8_finalize_network_activation_calibration(net);
#endif
		reset_wait_stream_events();
	}

	// The first eager warmup may turn provisional links into active FP8 relays
	// or cut a saturated edge to the BF16/FP32 fallback.  Count the resulting
	// executable graph, not merely the pre-calibration plan.
	fp8_eligible = 0;
	fp8_relay_enabled_links = 0;
	for (int idx = 0; idx < net.n; ++idx)
	{
		const Darknet::Layer & layer = net.layers[idx];
		fp8_eligible += layer.fp8_eligible != 0;
		fp8_relay_enabled_links += net.fp8_inference &&
			layer.type == Darknet::ELayerType::CONVOLUTIONAL && layer.fp8_relay_enabled != 0;
	}
	const unsigned long long fp8_relay_before_timing = net.fp8_activation_relay_executions;
	const unsigned long long fp8_graph_fused_before_timing = net.fp8_graph_fused_executions;

#ifdef DARKNET_HAS_FP4
	const Fp4DispatchCounts fp4_dispatch_before_timing = fp4_dispatch_counts(net);
#endif

	cudaEvent_t start = nullptr;
	cudaEvent_t stop = nullptr;
	CHECK_CUDA(cudaEventCreate(&start));
	CHECK_CUDA(cudaEventCreate(&stop));

	std::vector<float> timings;
	timings.reserve(static_cast<size_t>(iterations));
	for (int idx = 0; idx < iterations; ++idx)
	{
		CHECK_CUDA(cudaEventRecord(start, stream));
		enqueue_forward();
		CHECK_CUDA(cudaEventRecord(stop, stream));
		CHECK_CUDA(cudaEventSynchronize(stop));

		float elapsed_ms = 0.0f;
		CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
		timings.push_back(elapsed_ms);
		reset_wait_stream_events();
	}

	const TimingSummary summary = summarize_timings(timings);
	const float images_per_second = summary.mean_ms > 0.0f ? (1000.0f * net.batch / summary.mean_ms) : 0.0f;
	const unsigned long long fp8_relay_executions = net.fp8_activation_relay_executions - fp8_relay_before_timing;
	const unsigned long long fp8_graph_fused_executions = net.fp8_graph_fused_executions - fp8_graph_fused_before_timing;

#ifdef DARKNET_HAS_FP4
	const Fp4DispatchCounts fp4_dispatch = fp4_dispatch_counts(net) - fp4_dispatch_before_timing;
#endif

	int device = 0;
	cudaDeviceProp properties = {};
	CHECK_CUDA(cudaGetDevice(&device));
	CHECK_CUDA(cudaGetDeviceProperties(&properties, device));

	std::ostringstream result;
	result << std::fixed << std::setprecision(4)
		<< "{\"benchmark\":\"precision-inference-device-only\""
		<< ",\"precision_requested\":\"" << requested << "\""
		<< ",\"precision_effective\":\"" << effective << "\""
		<< ",\"device\":\"" << properties.name << "\""
		<< ",\"compute_capability\":\"" << properties.major << "." << properties.minor << "\""
		<< ",\"width\":" << net.w
		<< ",\"height\":" << net.h
		<< ",\"channels\":" << net.c
		<< ",\"batch\":" << net.batch
		<< ",\"warmup\":" << warmup
		<< ",\"iterations\":" << iterations
		<< ",\"mean_ms\":" << summary.mean_ms
		<< ",\"median_ms\":" << summary.median_ms
		<< ",\"p95_ms\":" << summary.p95_ms
		<< ",\"images_per_second\":" << images_per_second
		<< ",\"fp4_eligible_layers\":" << fp4_eligible
		<< ",\"fp4_prepacked_weight_layers\":" << fp4_prepacked
		<< ",\"fp4_relay_links\":" << fp4_relay_links
		<< ",\"fp8_eligible_layers\":" << fp8_eligible
		<< ",\"fp8_prepacked_weight_layers\":" << fp8_prepacked
		<< ",\"precision_planned_fp8_layers\":" << net.precision_planned_fp8_layers
		<< ",\"precision_planned_fp4_layers\":" << net.precision_planned_fp4_layers
		<< ",\"bf16_fallback_layers\":" << net.precision_bf16_fallback_layers
		<< ",\"fp8_activation_calibration_samples\":" << net.fp8_activation_calibration_samples
		<< ",\"fp8_relay_links\":" << fp8_relay_links
		<< ",\"fp8_relay_enabled_links\":" << fp8_relay_enabled_links
		<< ",\"fp8_relay_executions\":" << fp8_relay_executions
		<< ",\"fp8_graph_fused_executions\":" << fp8_graph_fused_executions
		<< ",\"fp8_relay_saturation_fallbacks\":" << net.fp8_activation_saturation_fallbacks
		<< ",\"synthetic_input\":true"
		<< ",\"weights_loaded\":" << (weights_loaded ? "true" : "false")
		<< ",\"synthetic_fp8_scales\":" << (synthetic_fp8_scales ? "true" : "false");
#ifdef DARKNET_HAS_FP4
	result << ",\"fp4_cublaslt_nvfp4_gemms\":" << fp4_dispatch.cublaslt
		<< ",\"fp4_relay_gemms\":" << fp4_dispatch.relay
		<< ",\"fp4_cudnn_frontend_fallback_gemms\":" << fp4_dispatch.cudnn_frontend
		<< ",\"fp4_failed_gemms\":" << fp4_dispatch.failed;
#else
	result << ",\"fp4_cublaslt_nvfp4_gemms\":0"
		<< ",\"fp4_relay_gemms\":0"
		<< ",\"fp4_cudnn_frontend_fallback_gemms\":0"
		<< ",\"fp4_failed_gemms\":0";
#endif
	result
		<< "}";
	*cfg_and_state.output << result.str() << std::endl;

#ifdef DARKNET_HAS_FP4
	if (requested == "fp4" && fp4_dispatch.cublaslt == 0)
	{
		CHECK_CUDA(cudaEventDestroy(stop));
		CHECK_CUDA(cudaEventDestroy(start));
		cuda_free(input_gpu);
		deactivate_network_streams();
		free_network(net);
		darknet_fatal_error(DARKNET_LOC,
			"FP4 was requested but no direct cuBLASLt NVFP4 GEMM executed during the timed benchmark");
	}
#endif

	CHECK_CUDA(cudaEventDestroy(stop));
	CHECK_CUDA(cudaEventDestroy(start));
	cuda_free(input_gpu);
	deactivate_network_streams();
	free_network(net);
#endif
}
