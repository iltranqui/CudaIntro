# Darknet object detection framework


# add_compile_definitions() requires 3.12
# CUDA detection requires 3.18
# runtime_dependencies requires 3.21 (windows only)
# cuda_architectures auto detection requires 3.24
CMAKE_MINIMUM_REQUIRED (VERSION 3.24)

INCLUDE (CM_version.cmake)

SET (CMAKE_CXX_STANDARD 23)
SET (CMAKE_CXX_STANDARD_REQUIRED ON)

IF (NOT DEFINED DARKNET_PREFERRED_GCC_VERSION)
	SET (DARKNET_PREFERRED_GCC_VERSION "14")
ENDIF ()

IF (NOT DEFINED CMAKE_C_COMPILER AND EXISTS "/usr/bin/gcc-${DARKNET_PREFERRED_GCC_VERSION}")
	SET (CMAKE_C_COMPILER "/usr/bin/gcc-${DARKNET_PREFERRED_GCC_VERSION}" CACHE FILEPATH "Preferred GNU C compiler for Darknet.")
ENDIF ()

IF (NOT DEFINED CMAKE_CXX_COMPILER AND EXISTS "/usr/bin/g++-${DARKNET_PREFERRED_GCC_VERSION}")
	SET (CMAKE_CXX_COMPILER "/usr/bin/g++-${DARKNET_PREFERRED_GCC_VERSION}" CACHE FILEPATH "Preferred GNU C++ compiler for Darknet.")
ENDIF ()

IF (NOT EXISTS "/usr/bin/gcc-${DARKNET_PREFERRED_GCC_VERSION}")
	MESSAGE (WARNING "gcc-${DARKNET_PREFERRED_GCC_VERSION} not found, falling back to system default compiler -- this can cause std:: link errors if it differs from other build machines.")
ENDIF ()
MESSAGE (STATUS "Darknet C compiler: ${CMAKE_C_COMPILER}")
MESSAGE (STATUS "Darknet C++ compiler: ${CMAKE_CXX_COMPILER}")

IF (NOT DEFINED DARKNET_PREFERRED_CUDA_ROOT)
	IF (EXISTS "/usr/local/cuda")
		SET (DARKNET_PREFERRED_CUDA_ROOT "/usr/local/cuda")
	ELSE ()
		FILE (GLOB darknet_cuda_versioned_roots LIST_DIRECTORIES TRUE "/usr/local/cuda-*")
		IF (darknet_cuda_versioned_roots)
			LIST (SORT darknet_cuda_versioned_roots COMPARE NATURAL ORDER DESCENDING)
			LIST (GET darknet_cuda_versioned_roots 0 DARKNET_PREFERRED_CUDA_ROOT)
		ENDIF ()
	ENDIF ()
ENDIF ()

IF (DARKNET_PREFERRED_CUDA_ROOT)
	IF (NOT DEFINED CUDAToolkit_ROOT)
		SET (CUDAToolkit_ROOT "${DARKNET_PREFERRED_CUDA_ROOT}" CACHE PATH "Preferred CUDA toolkit root for Darknet.")
	ENDIF ()

	IF (NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS "${DARKNET_PREFERRED_CUDA_ROOT}/bin/nvcc")
		SET (CMAKE_CUDA_COMPILER "${DARKNET_PREFERRED_CUDA_ROOT}/bin/nvcc" CACHE FILEPATH "Preferred CUDA compiler for Darknet.")
	ENDIF ()
ENDIF ()


IF (NOT CMAKE_BUILD_TYPE)
	# create either Release or Debug builds
	SET (CMAKE_BUILD_TYPE Release)
ENDIF ()

# to get verbose build output, set this to ON
SET (CMAKE_VERBOSE_MAKEFILE OFF)

# to get lots of debug output while running cmake, set this to ON
SET (CMAKE_FIND_DEBUG_MODE OFF)

PROJECT (Darknet LANGUAGES C CXX VERSION ${DARKNET_VERSION_SHORT})

INCLUDE (CheckLanguage)
INCLUDE (CMakeDependentOption)

INCLUDE (CM_misc.cmake)
INCLUDE (CM_dependencies.cmake)
INCLUDE (CM_source.cmake)
INCLUDE (CM_package.cmake)

# If you have Google Test installed, then you can enable the unit tests.
# On Ubuntu for example, you'd run the following commands:
#
#		sudo apt-get install googletest
#		cd /usr/src/googletest/
#		sudo mkdir build
#		cd build
#		sudo cmake ..
#		sudo make
#		sudo cp lib/*.a /usr/lib
#
# Once you successfully built Darknet, run "make test", "ctest", or "darknet_tests".
#
ENABLE_TESTING ()
ADD_TEST (
	NAME darknet_precision_architectures
	COMMAND ${CMAKE_COMMAND} -P ${CMAKE_SOURCE_DIR}/cmake/tests/test_precision_architectures.cmake)

# Google Test support is disabled for now (removed from the build).
# To re-enable: install gtest and uncomment the two lines below.
# FIND_LIBRARY (GTEST gtest)
# IF (GTEST)
# 	ADD_SUBDIRECTORY (src-test)
# ENDIF ()
