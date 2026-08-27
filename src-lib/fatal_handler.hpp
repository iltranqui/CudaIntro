#pragma once

/** @file
 * Universal fatal-error handling for Darknet.
 *
 * Covers four failure modes that darknet_fatal_error() alone cannot reach:
 *   1. Normal code paths          -> darknet_fatal_error() (existing, unchanged)
 *   2. Unhandled C++ exceptions   -> std::set_terminate hook
 *   3. OS signals (SIGSEGV etc.)  -> sigaction handler
 *   4. CUDA errors not checked    -> CUDA_CHECK macro
 *   5. Hangs / infinite loops     -> watchdog thread
 *
 * Usage:
 *   Call Darknet::install_fatal_handlers() once at the top of main(),
 *   before any Darknet or CUDA code runs.  Wrap main()'s body in
 *   try/catch(const std::exception&) to catch darknet_fatal_error throws.
 *
 * @since 2026-06-05
 */

#include "darknet_internal.hpp"

#include <atomic>
#include <cstdint>


namespace Darknet
{
	/** Install all fatal-error hooks.
	 * Call once at the start of main() before any other Darknet code.
	 * @param watchdog_timeout_sec  Seconds of no heartbeat before the watchdog aborts.
	 *                              Pass 0 to disable the watchdog.
	 */
	void install_fatal_handlers(int watchdog_timeout_sec = 30);

	/** Bump the watchdog heartbeat counter.
	 * Call this once per training iteration / inference frame in the hot loop.
	 * Thread-safe; zero overhead (single relaxed atomic increment).
	 */
	inline void heartbeat() noexcept;

	/** Low-level fatal: log to stderr + abort().
	 * Safe to call from signal handlers and std::terminate.
	 * Does NOT throw.  Prefer darknet_fatal_error() everywhere else.
	 */
	[[noreturn]] void darknet_fatal_abort(
		const char * filename,
		const char * funcname,
		int          line,
		const char * msg) noexcept;
}


// ---- implementation of the inline ----------------------------------------

namespace Darknet
{
	namespace detail
	{
		extern std::atomic<std::uint64_t> g_heartbeat;
	}

	inline void heartbeat() noexcept
	{
		detail::g_heartbeat.fetch_add(1, std::memory_order_relaxed);
	}
}


// ---- CUDA_CHECK macro -------------------------------------------------------

/** Wrap every CUDA runtime call with this macro.
 * On error it calls darknet_fatal_error() with the CUDA error string,
 * so the throw propagates normally back to main()'s catch block.
 *
 * Example:
 *   CUDA_CHECK(cudaMalloc(&ptr, size));
 */
#ifdef __CUDACC__
#include <cuda_runtime.h>
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t _cuda_err = (call);                                   \
        if (_cuda_err != cudaSuccess)                                            \
            darknet_fatal_error(DARKNET_LOC,                                    \
                "CUDA error in " #call ": %s",                                  \
                cudaGetErrorString(_cuda_err));                                  \
    } while (0)
#endif
