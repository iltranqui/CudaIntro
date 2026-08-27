#pragma once

#ifdef DARKNET_HAS_FP8

#include "darknet_internal.hpp"

namespace Darknet
{
	/// Free a raw cudaMalloc()'d pointer owned by a layer and clear it.
	template <typename T>
	void fp8_release_device_ptr(T *& ptr)
	{
		if (ptr)
		{
			CHECK_CUDA(cudaFree(ptr));
			ptr = nullptr;
		}
	}

	/// Free a cuda_free()-managed pointer owned by a layer and clear it.
	template <typename T>
	void fp8_release_cuda_alloc(T *& ptr)
	{
		if (ptr)
		{
			cuda_free(ptr);
			ptr = nullptr;
		}
	}

	/// Destroy a type-erased plan pointer (Fp8GemmPlan* / Fp8ConvPlan* stored as void*) and clear it.
	template <typename PlanT>
	void fp8_release_plan(void *& ptr, void (*destroy)(PlanT *))
	{
		if (ptr)
		{
			destroy(static_cast<PlanT *>(ptr));
			ptr = nullptr;
		}
	}
}

#endif
