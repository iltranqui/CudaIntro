/*
 * Standalone cuDNN frontend FP8 convolution engine probe.
 *
 * This is intentionally independent from Darknet runtime code.  It answers the
 * go/no-go question for cuDNN 9 graph FP8 implicit convolution on the current
 * GPU before the main convolution path is modified.
 */

#include <cuda_runtime.h>
#include <cudnn_frontend.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fe = cudnn_frontend;

namespace {

struct CudaBuffer {
    void *ptr = nullptr;
    size_t bytes = 0;

    explicit CudaBuffer(size_t requested_bytes) : bytes(std::max<size_t>(requested_bytes, 1)) {
        auto status = cudaMalloc(&ptr, bytes);
        if (status != cudaSuccess) {
            std::ostringstream ss;
            ss << "cudaMalloc(" << bytes << ") failed: " << cudaGetErrorString(status);
            throw std::runtime_error(ss.str());
        }
        status = cudaMemset(ptr, 0, bytes);
        if (status != cudaSuccess) {
            std::ostringstream ss;
            ss << "cudaMemset(" << bytes << ") failed: " << cudaGetErrorString(status);
            throw std::runtime_error(ss.str());
        }
    }

    ~CudaBuffer() {
        if (ptr) {
            cudaFree(ptr);
        }
    }

    CudaBuffer(CudaBuffer const &) = delete;
    CudaBuffer &operator=(CudaBuffer const &) = delete;
};

struct CudnnHandle {
    cudnnHandle_t handle = nullptr;

    CudnnHandle() {
        auto status = cudnnCreate(&handle);
        if (status != CUDNN_STATUS_SUCCESS) {
            std::ostringstream ss;
            ss << "cudnnCreate failed: " << cudnnGetErrorString(status);
            throw std::runtime_error(ss.str());
        }
    }

    ~CudnnHandle() {
        if (handle) {
            cudnnDestroy(handle);
        }
    }

    CudnnHandle(CudnnHandle const &) = delete;
    CudnnHandle &operator=(CudnnHandle const &) = delete;
};

struct ConvCase {
    std::string name;
    int64_t n = 1;
    int64_t c = 1;
    int64_t h = 1;
    int64_t w = 1;
    int64_t k = 1;
    int64_t r = 1;
    int64_t s = 1;
    int64_t pad_h = 0;
    int64_t pad_w = 0;
    int64_t stride_h = 1;
    int64_t stride_w = 1;
    int64_t dilation_h = 1;
    int64_t dilation_w = 1;
    fe::DataType_t x_type = fe::DataType_t::FP8_E4M3;
    fe::DataType_t w_type = fe::DataType_t::FP8_E4M3;
    fe::DataType_t y_type = fe::DataType_t::FLOAT;
    bool fuse_bias = false;
    bool fuse_relu = false;
};

int64_t conv_out_dim(int64_t input, int64_t pad, int64_t dilation, int64_t filter, int64_t stride) {
    return (input + 2 * pad - dilation * (filter - 1) - 1) / stride + 1;
}

std::vector<int64_t> nhwc_stride(int64_t n, int64_t c, int64_t h, int64_t w) {
    (void)n;
    return {h * w * c, 1, w * c, c};
}

size_t dtype_size(fe::DataType_t type) {
    switch (type) {
    case fe::DataType_t::FP8_E4M3:
    case fe::DataType_t::FP8_E5M2:
        return 1;
    case fe::DataType_t::HALF:
    case fe::DataType_t::BFLOAT16:
        return 2;
    case fe::DataType_t::FLOAT:
        return 4;
    default:
        return 4;
    }
}

std::string dtype_name(fe::DataType_t type) {
    switch (type) {
    case fe::DataType_t::FP8_E4M3:
        return "FP8_E4M3";
    case fe::DataType_t::FP8_E5M2:
        return "FP8_E5M2";
    case fe::DataType_t::HALF:
        return "HALF";
    case fe::DataType_t::BFLOAT16:
        return "BFLOAT16";
    case fe::DataType_t::FLOAT:
        return "FLOAT";
    default:
        return "UNKNOWN";
    }
}

bool report_status(std::string const &stage, fe::error_t status) {
    if (status.is_good()) {
        std::cout << "  " << std::left << std::setw(26) << stage << "OK\n";
        return true;
    }
    std::cout << "  " << std::left << std::setw(26) << stage << "FAIL";
    auto message = status.get_message();
    if (!message.empty()) {
        std::cout << " - " << message;
    }
    std::cout << "\n";
    return false;
}

void print_case_header(std::string const &op, ConvCase const &c) {
    const auto p = conv_out_dim(c.h, c.pad_h, c.dilation_h, c.r, c.stride_h);
    const auto q = conv_out_dim(c.w, c.pad_w, c.dilation_w, c.s, c.stride_w);
    std::cout << "\n[" << op << "] " << c.name << "\n"
              << "  x=NCHW(" << c.n << "," << c.c << "," << c.h << "," << c.w << ") as NHWC-stride"
              << "  w=KCRS(" << c.k << "," << c.c << "," << c.r << "," << c.s << ") as KRSC-stride"
              << "  y=NCHW(" << c.n << "," << c.k << "," << p << "," << q << ") as NHWC-stride\n"
              << "  pad=" << c.pad_h << "x" << c.pad_w << " stride=" << c.stride_h << "x" << c.stride_w
              << " dilation=" << c.dilation_h << "x" << c.dilation_w << " types=" << dtype_name(c.x_type)
              << "/" << dtype_name(c.w_type) << " -> " << dtype_name(c.y_type)
              << " bias=" << (c.fuse_bias ? "yes" : "no") << " relu=" << (c.fuse_relu ? "yes" : "no") << "\n";
}

bool finalize_graph(fe::graph::Graph &graph, cudnnHandle_t handle, bool print_graph) {
    if (!report_status("validate", graph.validate())) {
        return false;
    }
    if (!report_status("build op graph", graph.build_operation_graph(handle))) {
        return false;
    }
    if (!report_status("heuristics A", graph.create_execution_plans({fe::HeurMode_t::A}))) {
        return false;
    }

    auto support = graph.check_support(handle);
    report_status("check support", support);
    if (support.is_bad()) {
        auto fallback = graph.create_execution_plans({fe::HeurMode_t::FALLBACK});
        report_status("fallback list", fallback);
        if (fallback.is_bad()) {
            return false;
        }
        support = graph.check_support(handle);
        report_status("fallback support", support);
        if (support.is_bad()) {
            return false;
        }
    }

    if (!report_status("build plan", graph.build_plans(handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE))) {
        return false;
    }

    std::string plan_name;
    if (graph.get_plan_name(plan_name).is_good()) {
        std::cout << "  selected plan            " << plan_name << "\n";
    }
    int64_t workspace_size = 0;
    if (graph.get_workspace_size(workspace_size).is_good()) {
        std::cout << "  workspace bytes          " << workspace_size << "\n";
    }
    if (print_graph) {
        std::cout << "  graph:\n" << graph.print() << "\n";
    }
    return true;
}

bool query_workspace(fe::graph::Graph &graph, int64_t &workspace_size) {
    auto status = graph.get_workspace_size(workspace_size);
    return report_status("workspace query", status);
}

void copy_float_to_device(CudaBuffer &buffer, std::vector<float> const &host) {
    auto status = cudaMemcpy(buffer.ptr, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        std::ostringstream ss;
        ss << "cudaMemcpy host->device failed: " << cudaGetErrorString(status);
        throw std::runtime_error(ss.str());
    }
}

bool validate_fprop_output(ConvCase const &c, CudaBuffer const &y_buffer, std::vector<float> const &bias_host) {
    const auto p = conv_out_dim(c.h, c.pad_h, c.dilation_h, c.r, c.stride_h);
    const auto q = conv_out_dim(c.w, c.pad_w, c.dilation_w, c.s, c.stride_w);
    const size_t output_count = static_cast<size_t>(c.n * c.k * p * q);

    if (c.y_type == fe::DataType_t::BFLOAT16 && !c.fuse_bias) {
        std::vector<uint16_t> host(output_count);
        auto status = cudaMemcpy(host.data(), y_buffer.ptr, host.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
        if (status != cudaSuccess) {
            std::cout << "  " << std::left << std::setw(26) << "reference check"
                      << "FAIL - cudaMemcpy device->host failed: " << cudaGetErrorString(status) << "\n";
            return false;
        }
        const bool all_zero = std::all_of(host.begin(), host.end(), [](uint16_t value) { return value == 0; });
        std::cout << "  " << std::left << std::setw(26) << "reference check" << (all_zero ? "OK\n" : "FAIL - nonzero BF16 output\n");
        return all_zero;
    }

    if (c.y_type != fe::DataType_t::FLOAT) {
        std::cout << "  " << std::left << std::setw(26) << "reference check" << "SKIP - output type not covered\n";
        return true;
    }

    std::vector<float> host(output_count);
    auto status = cudaMemcpy(host.data(), y_buffer.ptr, host.size() * sizeof(float), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        std::cout << "  " << std::left << std::setw(26) << "reference check"
                  << "FAIL - cudaMemcpy device->host failed: " << cudaGetErrorString(status) << "\n";
        return false;
    }

    float max_error = 0.0f;
    for (size_t idx = 0; idx < output_count; ++idx) {
        const int64_t k = static_cast<int64_t>(idx % static_cast<size_t>(c.k));
        float expected = c.fuse_bias ? bias_host[static_cast<size_t>(k)] : 0.0f;
        if (c.fuse_relu) {
            expected = std::max(expected, 0.0f);
        }
        max_error = std::max(max_error, std::fabs(host[idx] - expected));
    }

    const bool ok = max_error <= 1.0e-5f;
    std::cout << "  " << std::left << std::setw(26) << "reference check"
              << (ok ? "OK" : "FAIL") << " max_error=" << max_error << "\n";
    return ok;
}

std::shared_ptr<fe::graph::Tensor_attributes> scalar_tensor(std::shared_ptr<fe::graph::Graph> const &graph,
                                                            std::string const &name) {
    return graph->tensor(fe::graph::Tensor_attributes()
                             .set_name(name)
                             .set_dim({1, 1, 1, 1})
                             .set_stride({1, 1, 1, 1})
                             .set_data_type(fe::DataType_t::FLOAT));
}

bool run_fprop(cudnnHandle_t handle, ConvCase const &c, bool execute, bool print_graph) {
    print_case_header("fprop", c);
    const auto p = conv_out_dim(c.h, c.pad_h, c.dilation_h, c.r, c.stride_h);
    const auto q = conv_out_dim(c.w, c.pad_w, c.dilation_w, c.s, c.stride_w);

    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::FLOAT)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto X = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("x")
                               .set_dim({c.n, c.c, c.h, c.w})
                               .set_stride(nhwc_stride(c.n, c.c, c.h, c.w))
                               .set_data_type(c.x_type));
    auto W = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("w")
                               .set_dim({c.k, c.c, c.r, c.s})
                               .set_stride({c.c * c.r * c.s, 1, c.s * c.c, c.c})
                               .set_data_type(c.w_type));

    auto conv = graph->conv_fprop(
        X,
        W,
        fe::graph::Conv_fprop_attributes()
            .set_padding({c.pad_h, c.pad_w})
            .set_stride({c.stride_h, c.stride_w})
            .set_dilation({c.dilation_h, c.dilation_w})
            .set_name("conv"));

    auto dx = scalar_tensor(graph, "descale_x");
    auto dw = scalar_tensor(graph, "descale_w");
    auto out = graph->pointwise(conv, dx, fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::MUL));
    out = graph->pointwise(out, dw, fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::MUL));

    std::shared_ptr<fe::graph::Tensor_attributes> bias;
    if (c.fuse_bias) {
        bias = graph->tensor(fe::graph::Tensor_attributes()
                                 .set_name("bias")
                                 .set_dim({1, c.k, 1, 1})
                                 .set_stride({c.k, 1, c.k, c.k})
                                 .set_data_type(fe::DataType_t::FLOAT));
        out = graph->pointwise(out, bias, fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD));
    }
    if (c.fuse_relu) {
        out = graph->pointwise(out, fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::RELU_FWD));
    }
    out->set_output(true).set_data_type(c.y_type).set_dim({c.n, c.k, p, q}).set_stride(nhwc_stride(c.n, c.k, p, q));

    if (!finalize_graph(*graph, handle, print_graph)) {
        return false;
    }

    if (!execute) {
        return true;
    }

    int64_t workspace_size = 0;
    if (!query_workspace(*graph, workspace_size)) {
        return false;
    }
    CudaBuffer x_buffer(static_cast<size_t>(c.n * c.c * c.h * c.w) * dtype_size(c.x_type));
    CudaBuffer w_buffer(static_cast<size_t>(c.k * c.c * c.r * c.s) * dtype_size(c.w_type));
    CudaBuffer y_buffer(static_cast<size_t>(c.n * c.k * p * q) * dtype_size(c.y_type));
    CudaBuffer descale_x(sizeof(float));
    CudaBuffer descale_w(sizeof(float));
    CudaBuffer workspace(static_cast<size_t>(workspace_size));
    copy_float_to_device(descale_x, {1.0f});
    copy_float_to_device(descale_w, {1.0f});

    std::vector<float> bias_host(static_cast<size_t>(c.k), 0.0f);
    std::unique_ptr<CudaBuffer> bias_buffer;
    if (bias) {
        bias_buffer = std::make_unique<CudaBuffer>(static_cast<size_t>(c.k) * sizeof(float));
        for (int64_t idx = 0; idx < c.k; ++idx) {
            bias_host[static_cast<size_t>(idx)] = static_cast<float>((idx % 7) - 3) * 0.25f;
        }
        copy_float_to_device(*bias_buffer, bias_host);
    }

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void *> variant_pack = {
        {X, x_buffer.ptr}, {W, w_buffer.ptr}, {out, y_buffer.ptr}, {dx, descale_x.ptr}, {dw, descale_w.ptr}};
    if (bias) {
        variant_pack.emplace(bias, bias_buffer->ptr);
    }

    if (!report_status("execute", graph->execute(handle, variant_pack, workspace.ptr))) {
        return false;
    }
    return validate_fprop_output(c, y_buffer, bias_host);
}

bool run_dgrad(cudnnHandle_t handle, ConvCase const &c, bool execute, bool print_graph) {
    print_case_header("dgrad", c);
    const auto p = conv_out_dim(c.h, c.pad_h, c.dilation_h, c.r, c.stride_h);
    const auto q = conv_out_dim(c.w, c.pad_w, c.dilation_w, c.s, c.stride_w);

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::FLOAT)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto DY = graph.tensor(fe::graph::Tensor_attributes()
                               .set_name("dy")
                               .set_dim({c.n, c.k, p, q})
                               .set_stride(nhwc_stride(c.n, c.k, p, q))
                               .set_data_type(fe::DataType_t::FP8_E5M2));
    auto W = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("w")
                              .set_dim({c.k, c.c, c.r, c.s})
                              .set_stride({c.c * c.r * c.s, 1, c.s * c.c, c.c})
                              .set_data_type(c.w_type));
    auto DX = graph.conv_dgrad(
        DY,
        W,
        fe::graph::Conv_dgrad_attributes()
            .set_padding({c.pad_h, c.pad_w})
            .set_stride({c.stride_h, c.stride_w})
            .set_dilation({c.dilation_h, c.dilation_w})
            .set_name("dgrad"));
    DX->set_dim({c.n, c.c, c.h, c.w})
        .set_stride(nhwc_stride(c.n, c.c, c.h, c.w))
        .set_output(true)
        .set_data_type(fe::DataType_t::FLOAT);

    if (!finalize_graph(graph, handle, print_graph)) {
        return false;
    }

    if (!execute) {
        return true;
    }

    int64_t workspace_size = 0;
    if (!query_workspace(graph, workspace_size)) {
        return false;
    }
    CudaBuffer dy_buffer(static_cast<size_t>(c.n * c.k * p * q) * dtype_size(fe::DataType_t::FP8_E5M2));
    CudaBuffer w_buffer(static_cast<size_t>(c.k * c.c * c.r * c.s) * dtype_size(c.w_type));
    CudaBuffer dx_buffer(static_cast<size_t>(c.n * c.c * c.h * c.w) * sizeof(float));
    CudaBuffer workspace(static_cast<size_t>(workspace_size));

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void *> variant_pack = {
        {DY, dy_buffer.ptr}, {W, w_buffer.ptr}, {DX, dx_buffer.ptr}};
    return report_status("execute", graph.execute(handle, variant_pack, workspace.ptr));
}

bool run_wgrad(cudnnHandle_t handle, ConvCase const &c, bool execute, bool print_graph) {
    print_case_header("wgrad", c);
    const auto p = conv_out_dim(c.h, c.pad_h, c.dilation_h, c.r, c.stride_h);
    const auto q = conv_out_dim(c.w, c.pad_w, c.dilation_w, c.s, c.stride_w);

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::FLOAT)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto X = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("x")
                              .set_dim({c.n, c.c, c.h, c.w})
                              .set_stride(nhwc_stride(c.n, c.c, c.h, c.w))
                              .set_data_type(c.x_type));
    auto DY = graph.tensor(fe::graph::Tensor_attributes()
                               .set_name("dy")
                               .set_dim({c.n, c.k, p, q})
                               .set_stride(nhwc_stride(c.n, c.k, p, q))
                               .set_data_type(fe::DataType_t::FP8_E5M2));
    auto DW = graph.conv_wgrad(
        DY,
        X,
        fe::graph::Conv_wgrad_attributes()
            .set_padding({c.pad_h, c.pad_w})
            .set_stride({c.stride_h, c.stride_w})
            .set_dilation({c.dilation_h, c.dilation_w})
            .set_name("wgrad"));
    DW->set_dim({c.k, c.c, c.r, c.s})
        .set_stride({c.c * c.r * c.s, 1, c.s * c.c, c.c})
        .set_output(true)
        .set_data_type(fe::DataType_t::FLOAT);

    if (!finalize_graph(graph, handle, print_graph)) {
        return false;
    }

    if (!execute) {
        return true;
    }

    int64_t workspace_size = 0;
    if (!query_workspace(graph, workspace_size)) {
        return false;
    }
    CudaBuffer x_buffer(static_cast<size_t>(c.n * c.c * c.h * c.w) * dtype_size(c.x_type));
    CudaBuffer dy_buffer(static_cast<size_t>(c.n * c.k * p * q) * dtype_size(fe::DataType_t::FP8_E5M2));
    CudaBuffer dw_buffer(static_cast<size_t>(c.k * c.c * c.r * c.s) * sizeof(float));
    CudaBuffer workspace(static_cast<size_t>(workspace_size));

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void *> variant_pack = {
        {X, x_buffer.ptr}, {DY, dy_buffer.ptr}, {DW, dw_buffer.ptr}};
    return report_status("execute", graph.execute(handle, variant_pack, workspace.ptr));
}

void print_device() {
    int device = 0;
    auto status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("cudaGetDevice failed: ") + cudaGetErrorString(status));
    }
    cudaDeviceProp prop{};
    status = cudaGetDeviceProperties(&prop, device);
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("cudaGetDeviceProperties failed: ") + cudaGetErrorString(status));
    }
    std::cout << "fp8_conv_probe\n"
              << "  cudnn runtime            " << cudnnGetVersion() << "\n"
              << "  cuda runtime             " << cudnnGetCudartVersion() << "\n"
              << "  device                   " << prop.name << " sm_" << prop.major << prop.minor << "\n";
}

}  // namespace

int main(int argc, char **argv) {
    bool execute = true;
    bool print_graph = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--no-execute") {
            execute = false;
        } else if (arg == "--print-graph") {
            print_graph = true;
        } else if (arg == "--help") {
            std::cout << "Usage: fp8_conv_probe [--no-execute] [--print-graph]\n";
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return 2;
        }
    }

    try {
        print_device();
        CudnnHandle handle;

        std::vector<ConvCase> cases = {
            {.name = "lego-3x3-s1-bf16out",
             .n = 64,
             .c = 64,
             .h = 40,
             .w = 56,
             .k = 64,
             .r = 3,
             .s = 3,
             .pad_h = 1,
             .pad_w = 1,
             .y_type = fe::DataType_t::BFLOAT16},
            {.name = "lego-3x3-s1-bias-relu-f32out",
             .n = 64,
             .c = 64,
             .h = 40,
             .w = 56,
             .k = 64,
             .r = 3,
             .s = 3,
             .pad_h = 1,
             .pad_w = 1,
             .fuse_bias = true,
             .fuse_relu = true},
            {.name = "stride2-1x1-f32out",
             .n = 64,
             .c = 64,
             .h = 40,
             .w = 56,
             .k = 64,
             .r = 1,
             .s = 1,
             .stride_h = 2,
             .stride_w = 2},
            {.name = "unaligned-c48-k48-3x3",
             .n = 32,
             .c = 48,
             .h = 32,
             .w = 32,
             .k = 48,
             .r = 3,
             .s = 3,
             .pad_h = 1,
             .pad_w = 1}};

        int fprop_ok = 0;
        int dgrad_ok = 0;
        int wgrad_ok = 0;
        for (auto const &c : cases) {
            fprop_ok += run_fprop(handle.handle, c, execute, print_graph) ? 1 : 0;
            dgrad_ok += run_dgrad(handle.handle, c, execute, print_graph) ? 1 : 0;
            wgrad_ok += run_wgrad(handle.handle, c, execute, print_graph) ? 1 : 0;
        }

        std::cout << "\nsummary\n"
                  << "  fprop supported          " << fprop_ok << "/" << cases.size() << "\n"
                  << "  dgrad supported          " << dgrad_ok << "/" << cases.size() << "\n"
                  << "  wgrad supported          " << wgrad_ok << "/" << cases.size() << "\n";

        return fprop_ok > 0 ? 0 : 1;
    } catch (std::exception const &e) {
        std::cerr << "fp8_conv_probe fatal: " << e.what() << "\n";
        return 1;
    }
}
