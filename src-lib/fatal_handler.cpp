#include "fatal_handler.hpp"
#include "darknet_internal.hpp"

#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <thread>
#include <chrono>

#include <execinfo.h>   // backtrace, backtrace_symbols_fd  (Linux / WSL)
#include <unistd.h>     // STDERR_FILENO, _exit
#include <pthread.h>    // pthread_self, pthread_kill
#include <fcntl.h>      // open, O_CREAT, O_APPEND


namespace Darknet
{
namespace detail
{
	std::atomic<std::uint64_t>  g_heartbeat{0};
	static std::atomic<pthread_t> g_main_tid{pthread_self()};

	// fd of "exception.log", opened once (non-signal context) by install_fatal_handlers().
	// -1 means "not open"; handlers must check before writing to it.
	static std::atomic<int> g_log_fd{-1};

	// -------------------------------------------------------------------------
	// Async-signal-safe helpers: duplicate every crash dump to stderr AND to
	// exception.log, using only write()/backtrace_symbols_fd() (no malloc, no
	// iostreams, no fopen — none of those are safe inside a signal handler).
	// -------------------------------------------------------------------------
	static void dual_write(const char * buf, std::size_t len) noexcept
	{
		(void)write(STDERR_FILENO, buf, len);

		const int fd = g_log_fd.load(std::memory_order_relaxed);
		if (fd >= 0)
		{
			(void)write(fd, buf, len);
		}
	}

	static void dual_backtrace(void * const * frames, int n) noexcept
	{
		backtrace_symbols_fd(frames, n, STDERR_FILENO);

		const int fd = g_log_fd.load(std::memory_order_relaxed);
		if (fd >= 0)
		{
			backtrace_symbols_fd(frames, n, fd);
		}
	}

	// -------------------------------------------------------------------------
	// Async-signal-safe stack dump + _exit.
	// Must NOT call malloc, throw, or any non-reentrant function.
	// -------------------------------------------------------------------------
	static void dump_and_exit(int signum) noexcept
	{
		// write a minimal header without printf (not async-signal-safe on all libc)
		const char msg[] = "\n[fatal_handler] signal received, stack trace:\n";
		dual_write(msg, sizeof(msg) - 1);

		void * frames[64];
		const int n = backtrace(frames, 64);
		dual_backtrace(frames, n);

		// _exit skips atexit/destructors; avoids re-entrancy into the C++ runtime
		_exit(signum + 128);
	}

	// -------------------------------------------------------------------------
	// std::terminate handler — called for unhandled C++ exceptions or
	// explicit std::terminate() calls.
	// -------------------------------------------------------------------------
	static void terminate_handler() noexcept
	{
		const char hdr[] = "\n[fatal_handler] std::terminate called\n";
		dual_write(hdr, sizeof(hdr) - 1);

		// try to extract the exception message while re-throwing.
		// std::terminate() runs on the throwing thread's normal stack (not
		// inside a signal handler), so snprintf/exceptions are safe here.
		const auto eptr = std::current_exception();
		if (eptr)
		{
			try { std::rethrow_exception(eptr); }
			catch (const std::exception & e)
			{
				char buf[1024];
				const int len = std::snprintf(buf, sizeof(buf),
					"[fatal_handler] unhandled exception: %s\n", e.what());
				if (len > 0)
				{
					const std::size_t wlen = (static_cast<std::size_t>(len) < sizeof(buf) - 1)
						? static_cast<std::size_t>(len) : sizeof(buf) - 1;
					dual_write(buf, wlen);
				}
			}
			catch (...)
			{
				const char uk[] = "[fatal_handler] unhandled exception: (unknown type)\n";
				dual_write(uk, sizeof(uk) - 1);
			}
		}

		void * frames[64];
		const int n = backtrace(frames, 64);
		dual_backtrace(frames, n);

		_exit(1);
	}

	// -------------------------------------------------------------------------
	// SIGUSR1 handler: dump the *receiving* thread's stack.
	// The watchdog sends this to the stuck thread, then itself calls abort().
	// -------------------------------------------------------------------------
	static void usr1_handler(int) noexcept
	{
		const char msg[] = "\n[fatal_handler] watchdog: stuck thread stack trace:\n";
		dual_write(msg, sizeof(msg) - 1);

		void * frames[64];
		const int n = backtrace(frames, 64);
		dual_backtrace(frames, n);
		// do NOT abort here — the watchdog thread does that after we flush
	}

	// -------------------------------------------------------------------------
	// Watchdog thread body.
	// -------------------------------------------------------------------------
	static void watchdog_body(int timeout_seconds)
	{
		uint64_t last   = g_heartbeat.load(std::memory_order_relaxed);
		int      stalled = 0;

		for (;;)
		{
			std::this_thread::sleep_for(std::chrono::seconds(1));

			const uint64_t now = g_heartbeat.load(std::memory_order_relaxed);
			if (now == last)
			{
				if (++stalled >= timeout_seconds)
				{
					std::fprintf(stderr,
						"\n[fatal_handler] watchdog: no heartbeat for %d seconds, "
						"dumping stuck thread and aborting\n",
						timeout_seconds);
					std::fflush(stderr);

					// signal the main/worker thread to print its own backtrace
					pthread_kill(g_main_tid.load(), SIGUSR1);
					std::this_thread::sleep_for(std::chrono::milliseconds(300));

					std::abort();   // SIGABRT → core dump (ulimit -c unlimited)
				}
			}
			else
			{
				stalled = 0;
				last    = now;
			}
		}
	}

} // namespace detail


// ---- public API -------------------------------------------------------------

[[noreturn]] void darknet_fatal_abort(
	const char * filename,
	const char * funcname,
	int          line,
	const char * msg) noexcept
{
	char buf[1024];
	const int len = std::snprintf(buf, sizeof(buf), "\n[FATAL] %s:%d (%s): %s\n",
		filename, line, funcname, msg);
	if (len > 0)
	{
		const std::size_t wlen = (static_cast<std::size_t>(len) < sizeof(buf) - 1)
			? static_cast<std::size_t>(len) : sizeof(buf) - 1;
		detail::dual_write(buf, wlen);
	}

	void * frames[64];
	const int n = backtrace(frames, 64);
	detail::dual_backtrace(frames, n);

	std::abort();
}


void install_fatal_handlers(const int watchdog_timeout_sec)
{
	// record the calling thread (main / worker) for the watchdog to signal
	detail::g_main_tid.store(pthread_self());

	// --- 0. Crash log --------------------------------------------------------
	// Open exception.log once, up front, in a normal (non-signal) context.
	// The raw fd is kept open for the life of the process and written to
	// directly with write()/backtrace_symbols_fd() from signal/terminate
	// handlers — opening a file from inside a handler is not async-signal-safe.
	{
		const int fd = ::open("exception.log", O_WRONLY | O_CREAT | O_APPEND, 0644);
		detail::g_log_fd.store(fd, std::memory_order_relaxed);
	}

	// --- 1. OS signals -------------------------------------------------------
	struct sigaction sa{};
	sa.sa_handler = detail::dump_and_exit;
	sigemptyset(&sa.sa_mask);
	sa.sa_flags = SA_RESETHAND;   // one-shot: let default handler fire on re-entry

	for (const int sig : {SIGSEGV, SIGFPE, SIGILL, SIGBUS})
	{
		sigaction(sig, &sa, nullptr);
	}

	// SIGUSR1 is used by the watchdog to dump the stuck thread's stack
	struct sigaction sa_usr1{};
	sa_usr1.sa_handler = detail::usr1_handler;
	sigemptyset(&sa_usr1.sa_mask);
	sa_usr1.sa_flags = 0;
	sigaction(SIGUSR1, &sa_usr1, nullptr);

	// --- 2. Unhandled C++ exceptions -----------------------------------------
	std::set_terminate(detail::terminate_handler);

	// --- 3. Watchdog (optional) ----------------------------------------------
	if (watchdog_timeout_sec > 0)
	{
		std::thread(detail::watchdog_body, watchdog_timeout_sec).detach();
	}
}

} // namespace Darknet
