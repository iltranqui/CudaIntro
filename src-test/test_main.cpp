#include <gtest/gtest.h>
#include <cstdlib>

#include "darknet.hpp"


int main(int argc, char **argv)
{
	testing::InitGoogleTest(&argc, argv);

	// GPU tests enabled by default; set DARKNET_TEST_GPU=0 to disable
	const char* gpu_mode = std::getenv("DARKNET_TEST_GPU");
	if (gpu_mode && std::string(gpu_mode) == "0")
	{
		Darknet::set_gpu_index(-1);  // CPU only
	}
	else
	{
		Darknet::set_gpu_index(0);  // Enable GPU for tests (default)
	}

	return RUN_ALL_TESTS();
}
