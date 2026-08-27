#include <gtest/gtest.h>

#include "darknet_internal.hpp"
#include "fp4_kernels.hpp"
#include "fp4_scaling.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace
{
	std::vector<uint8_t> stochastic_reference(const std::vector<float> & input, const uint64_t seed)
	{
		std::vector<uint8_t> packed(Darknet::fp4_packed_byte_count(input.size()), 0U);
		for (size_t index = 0; index < input.size(); ++index)
		{
			const uint8_t nibble = Darknet::fp4_encode_e2m1_stochastic(input[index], seed, index);
			packed[index / 2U] |= static_cast<uint8_t>(nibble << ((index & 1U) * 4U));
		}
		return packed;
	}

	std::vector<uint8_t> run_gpu(const std::vector<float> & input, const bool stochastic, const uint64_t seed)
	{
		float * input_gpu = nullptr;
		uint8_t * packed_gpu = nullptr;
		const size_t packed_bytes = Darknet::fp4_packed_byte_count(input.size());
		CHECK_CUDA(cudaMalloc(&input_gpu, input.size() * sizeof(float)));
		CHECK_CUDA(cudaMalloc(&packed_gpu, packed_bytes));
		CHECK_CUDA(cudaMemcpyAsync(input_gpu, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice, get_cuda_stream()));
		CHECK_CUDA(cudaMemsetAsync(packed_gpu, 0xa5, packed_bytes, get_cuda_stream()));
		if (stochastic)
			Darknet::fp4_pack_e2m1_stochastic_gpu(input_gpu, input.size(), seed, packed_gpu);
		else
			Darknet::fp4_pack_e2m1_gpu(input_gpu, input.size(), packed_gpu);
		std::vector<uint8_t> result(packed_bytes);
		CHECK_CUDA(cudaMemcpyAsync(result.data(), packed_gpu, packed_bytes, cudaMemcpyDeviceToHost, get_cuda_stream()));
		CHECK_CUDA(cudaStreamSynchronize(get_cuda_stream()));
		CHECK_CUDA(cudaFree(packed_gpu));
		CHECK_CUDA(cudaFree(input_gpu));
		return result;
	}
}

TEST(Fp4Kernels, NearestEvenPackingMatchesCpuReferenceIncludingOddTail)
{
	const std::vector<float> input = {
		-std::numeric_limits<float>::infinity(), -6.0f, -3.5f, -0.0f,
		0.0f, 0.25f, 0.75f, 1.75f, 3.5f, 6.0f,
		std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN(), 1.6f};
	EXPECT_EQ(run_gpu(input, false, 0U), Darknet::fp4_pack_e2m1(input.data(), input.size()));
	EXPECT_EQ(run_gpu(input, false, 0U).back() & 0xf0U, 0U);
}

TEST(Fp4Kernels, NearestEvenPackingMatchesCpuAcrossManyBlocks)
{
	std::vector<float> input(4099U);
	for (size_t index = 0; index < input.size(); ++index)
		input[index] = std::sin(static_cast<float>(index) * 0.173f) * 7.25f;
	EXPECT_EQ(run_gpu(input, false, 0U), Darknet::fp4_pack_e2m1(input.data(), input.size()));
}

TEST(Fp4Kernels, SeededStochasticPackingIsDeterministicAndMatchesCpu)
{
	std::vector<float> input(4097U);
	for (size_t index = 0; index < input.size(); ++index)
		input[index] = std::sin(static_cast<float>(index) * 0.071f) * 5.875f;
	constexpr uint64_t seed = 0x123456789abcdef0ULL;
	const auto expected = stochastic_reference(input, seed);
	const auto first = run_gpu(input, true, seed);
	const auto second = run_gpu(input, true, seed);
	EXPECT_EQ(first, expected);
	EXPECT_EQ(second, expected);
	EXPECT_NE(first, run_gpu(input, true, seed + 1U));
	EXPECT_EQ(first.back() & 0xf0U, 0U);
}
