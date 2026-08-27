# Darknet object detection framework


# If FP4 was requested and this failure is fatal, error out; otherwise silently fall back to no FP4.
MACRO (darknet_disable_fp4 msg)
	IF (DARKNET_BUILD_FP4)
		IF (DARKNET_ENABLE_FP4 STREQUAL "ON")
			MESSAGE (FATAL_ERROR "${msg}")
		ENDIF ()
		SET (DARKNET_BUILD_FP4 OFF)
	ENDIF ()
ENDMACRO ()


# =================
# == NVIDIA CUDA ==
# =================
CMAKE_DEPENDENT_OPTION (DARKNET_TRY_CUDA "Attempt to find NVIDIA/CUDA GPU support" ON "" ON)
IF (DARKNET_TRY_CUDA)
	CHECK_LANGUAGE (CUDA)
	IF (CMAKE_CUDA_COMPILER)
		MESSAGE (STATUS "CUDA detected. Darknet will use NVIDIA GPUs.  CUDA compiler is ${CMAKE_CUDA_COMPILER}.")
		ENABLE_LANGUAGE (CUDA)
		FIND_PACKAGE(CUDAToolkit REQUIRED)
		MESSAGE (STATUS "CUDA toolkit ${CUDAToolkit_VERSION} found in ${CUDAToolkit_TARGET_DIR}.")
		INCLUDE_DIRECTORIES (${CUDAToolkit_INCLUDE_DIRS})
		ADD_COMPILE_DEFINITIONS (DARKNET_GPU_CUDA)
		ADD_COMPILE_DEFINITIONS (DARKNET_GPU)
		SET (CMAKE_CUDA_STANDARD 17)
		SET (CMAKE_CUDA_STANDARD_REQUIRED ON)

		# -- CUDA architectures --
		#
		# Build the installed GPU by default.  Explicit architecture lists remain
		# available for cross/fat binaries (for example 89-real;100-real;120-real).
		#
		# (note that Kepler sm_30 and sm_35 are no longer supported by newer CUDA toolkits)
		#
		#	50: Tesla Quadro M
		#	52: Quadro M6000, GeForce 900, 970, 980, Titan X
		#	53: Tegra Jetson TX1, X1, Drive CX, Drive PX, Jetson Nano
		#	60: Quadro GP100, Tesla P100, DGX-1
		#	61: GTX 1080, 1070, 1060, 1050, 1030, 1010, GP108 Titan Xp, Tesla P40, Tesla P4, Drive PX2
		#	62: Drive PX2, Tegra Jetson TX2
		#	70: DGX-1 Volta, Tesla V100, GTX 1180 GV104, Titan V, Quadro VG100
		#	72: Jetson AGX Xavier, AGX Pegasus, Jetson Xavier NX
		#	75: GTX RTX Turing, GTX 1660, RTX 2060, RTX 2070, RTX 2080, Titan RTX, Quadro RTX 4000, 5000, 6000, 8000, T1000, T2000, Tesla T4, XNOR Tensor Cores
		#	80: A100, GA100, DGX-A100, RTX 3080 (?)
		#	86: Tesla GA10x, RTX Ampere, RTX 3050, 3070, 3080, 3090, GA102, GA107, RTX A2000, A3000, A4000, A5000, A6000, A40, GA106, RTX 3060, GA104, A10, A16, A40, A2 Tensor
		#	87: Jetson AGX Orin, Drive AGX Orin
		#	89: RTX 4090, 4080, 6000, Tesla L40
		#	90: H100, GH100
		#
		IF (NOT DEFINED DARKNET_CUDA_ARCHITECTURES)
			SET (DARKNET_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures emitted by Darknet")
		ENDIF ()
		set(DARKNET_ENABLE_FP8 "AUTO" CACHE STRING "Build FP8 sources: AUTO, ON, or OFF")
		set_property(CACHE DARKNET_ENABLE_FP8 PROPERTY STRINGS AUTO ON OFF)
		set(DARKNET_ENABLE_FP4 "AUTO" CACHE STRING "Build Blackwell FP4 sources: AUTO, ON, or OFF")
		set_property(CACHE DARKNET_ENABLE_FP4 PROPERTY STRINGS AUTO ON OFF)

		MESSAGE (STATUS "FP8 needs compute capability >= 89 and CUDA >= 12.1; FP4 needs compute capability >= 100 (Blackwell) and CUDA >= 13.2.")

		include("${CMAKE_SOURCE_DIR}/cmake/DarknetPrecisionArchitectures.cmake")
		set(DARKNET_NATIVE_COMPUTE_CAPABILITY "" CACHE STRING "Numeric CUDA compute capability used to resolve 'native' when GPU discovery is unavailable")
		set(darknet_native_compute_capability "")
		if("native" IN_LIST DARKNET_CUDA_ARCHITECTURES)
			if(DARKNET_NATIVE_COMPUTE_CAPABILITY MATCHES "^[0-9]+(\\;[0-9]+)*$")
				set(darknet_native_compute_capability "${DARKNET_NATIVE_COMPUTE_CAPABILITY}")
			else()
				execute_process(
				COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
				OUTPUT_VARIABLE darknet_native_compute_capability_text
				OUTPUT_STRIP_TRAILING_WHITESPACE
				ERROR_QUIET
				RESULT_VARIABLE darknet_nvidia_smi_result)
			endif()
			if(NOT darknet_native_compute_capability AND darknet_nvidia_smi_result EQUAL 0)
				string(REPLACE "\r" "" darknet_native_compute_capability_text "${darknet_native_compute_capability_text}")
				string(REPLACE "\n" ";" darknet_native_compute_capability_lines "${darknet_native_compute_capability_text}")
				foreach(darknet_compute_capability IN LISTS darknet_native_compute_capability_lines)
					string(STRIP "${darknet_compute_capability}" darknet_compute_capability)
					if(NOT darknet_compute_capability MATCHES "^[0-9]+\\.[0-9]+$")
						message(FATAL_ERROR "nvidia-smi returned malformed compute capability '${darknet_compute_capability}'")
					endif()
					string(REPLACE "." "" darknet_compute_capability "${darknet_compute_capability}")
					list(APPEND darknet_native_compute_capability "${darknet_compute_capability}")
				endforeach()
				list(REMOVE_DUPLICATES darknet_native_compute_capability)
			endif()
			if(NOT darknet_native_compute_capability)
				message(FATAL_ERROR "DARKNET_CUDA_ARCHITECTURES=native requires an accessible NVIDIA GPU; use an explicit architecture for cross-compilation")
			endif()
		endif()
		darknet_resolve_precision_architectures(
			"${DARKNET_CUDA_ARCHITECTURES}" "${darknet_native_compute_capability}"
			darknet_arch_has_fp8 darknet_arch_has_fp4 darknet_arch_has_sm89 DARKNET_RESOLVED_CUDA_ARCHITECTURES
			DARKNET_FP8_CUDA_ARCHITECTURES DARKNET_FP4_CUDA_ARCHITECTURES)
		set(DARKNET_FP8_TARGET_SM89 ${darknet_arch_has_sm89})
		set(DARKNET_CUDA_ARCHITECTURES "${DARKNET_RESOLVED_CUDA_ARCHITECTURES}")

		# -- FP8 support detection --
		set(darknet_fp8_api_available OFF)
		if(TARGET CUDA::cublasLt)
			include(CheckCXXSourceCompiles)
			set(CMAKE_REQUIRED_INCLUDES ${CUDAToolkit_INCLUDE_DIRS})
			check_cxx_source_compiles("#include <cuda_fp8.h>\n#include <cuda_runtime_api.h>\n#include <cublasLt.h>\nint main() { cudaDataType_t t = CUDA_R_8F_E4M3; auto a = CUBLASLT_MATMUL_DESC_A_SCALE_POINTER; auto b = CUBLASLT_MATMUL_DESC_B_SCALE_POINTER; return int(t) + int(a) + int(b); }" DARKNET_CUDA_HAS_FP8_API)
			unset(CMAKE_REQUIRED_INCLUDES)
			if(DARKNET_CUDA_HAS_FP8_API)
				set(darknet_fp8_api_available ON)
			endif()
		endif()
		if(darknet_arch_has_fp8 AND NOT darknet_fp8_api_available)
			if(DARKNET_ENABLE_FP8 STREQUAL "ON")
				message(FATAL_ERROR "FP8 disabled: requires compute capability >= 89 and CUDA >= 12.1")
			endif()
			set(darknet_arch_has_fp8 OFF)
		endif()

		# -- FP4 support detection --
		# Blackwell NVFP4 is intentionally pinned to the CUDA 13.2 toolchain used by
		# the supported build recipe.  Earlier toolkits expose some FP4 declarations,
		# but do not provide the ABI/runtime contract this target is tested against.
		set(darknet_fp4_api_available OFF)
		if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 13.2 AND TARGET CUDA::cublasLt)
			include(CheckCXXSourceCompiles)
			set(CMAKE_REQUIRED_INCLUDES ${CUDAToolkit_INCLUDE_DIRS})
			check_cxx_source_compiles("#include <cuda_fp4.h>\n#include <cuda_runtime_api.h>\n#include <cublasLt.h>\nint main() { cudaDataType_t t = CUDA_R_4F_E2M1; auto s = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3; return int(t) + int(s); }" DARKNET_CUDA_HAS_FP4_BLOCK_SCALE_API)
			unset(CMAKE_REQUIRED_INCLUDES)
			if(DARKNET_CUDA_HAS_FP4_BLOCK_SCALE_API)
				set(darknet_fp4_api_available ON)
			endif()
		endif()
		if(darknet_arch_has_fp4 AND NOT darknet_fp4_api_available)
			if(DARKNET_ENABLE_FP4 STREQUAL "ON")
				message(FATAL_ERROR "FP4 disabled: requires compute capability >= 100 (Blackwell) and CUDA >= 13.2")
			endif()
			set(darknet_arch_has_fp4 OFF)
		endif()
		darknet_resolve_precision_feature("${DARKNET_ENABLE_FP8}" "${darknet_arch_has_fp8}" DARKNET_ENABLE_FP8 DARKNET_BUILD_FP8)
		darknet_resolve_precision_feature("${DARKNET_ENABLE_FP4}" "${darknet_arch_has_fp4}" DARKNET_ENABLE_FP4 DARKNET_BUILD_FP4)
		message(STATUS "Darknet CUDA architectures: ${DARKNET_CUDA_ARCHITECTURES} (resolved: ${DARKNET_RESOLVED_CUDA_ARCHITECTURES})")

		# -- link libraries --
		SET (DARKNET_USE_CUDA ON)
		LIST (APPEND DARKNET_LINK_LIBS CUDA::cudart CUDA::cuda_driver CUDA::cublas CUDA::curand)
		IF (TARGET CUDA::cublasLt)
			LIST (APPEND DARKNET_LINK_LIBS CUDA::cublasLt)
		ENDIF ()
		IF (TARGET CUDA::nvrtc)
			# cudnn-frontend's runtime-compiled engines (compile_and_load_kernel) call into
			# NVRTC directly; without this, linking darknet/fp8_conv_probe fails with
			# undefined references to nvrtcCreateProgram et al.
			LIST (APPEND DARKNET_LINK_LIBS CUDA::nvrtc)
		ENDIF ()
	ELSE ()
		MESSAGE (WARNING "Support for NVIDIA CUDA not found.")
	ENDIF ()
ELSE ()
	MESSAGE (WARNING "Support for NVIDIA CUDA is disabled.")
ENDIF ()


# ===========
# == cuDNN ==
# ===========
IF (DARKNET_USE_CUDA)
	# -- cuDNN library discovery --
	# Look for cudnn, we will look in the same place as other CUDA libraries and also a few other places as well.
	FILE (GLOB darknet_cuda_versioned_roots LIST_DIRECTORIES TRUE "/usr/local/cuda-*")
	SET (darknet_cuda_search_roots
		${CUDAToolkit_TARGET_DIR}
		${CUDAToolkit_LIBRARY_ROOT}
		${CUDAToolkit_ROOT}
		${CUDAToolkit_LIBRARY_DIR}
		/usr/local/cuda
		${darknet_cuda_versioned_roots}
	)
	LIST (REMOVE_DUPLICATES darknet_cuda_search_roots)
	FIND_PATH(cudnn_include cudnn.h
				HINTS ${CUDAToolkit_INCLUDE_DIRS} ${darknet_cuda_search_roots} ENV CUDNN_INCLUDE_DIR ENV CUDA_PATH ENV CUDNN_HOME
				PATHS /usr/include/x86_64-linux-gnu /usr/include /usr/local /usr/local/cuda ENV CPATH
				PATH_SUFFIXES include targets/x86_64-linux/include)
	FIND_LIBRARY(cudnn cudnn
				HINTS ${CUDAToolkit_LIBRARY_DIR} ${darknet_cuda_search_roots} ENV CUDNN_LIBRARY_DIR ENV CUDA_PATH ENV CUDNN_HOME
				PATHS /usr/local /usr/local/cuda ENV LD_LIBRARY_PATH
				PATH_SUFFIXES lib64 lib/x64 lib x64 targets/x86_64-linux/lib targets/x86_64-linux/lib/stubs)
	IF (cudnn_include MATCHES "^/mnt/[a-zA-Z]/" AND EXISTS "/usr/include/x86_64-linux-gnu/cudnn.h")
		MESSAGE (STATUS "Ignoring Windows cuDNN include from WSL path: ${cudnn_include}")
		SET (cudnn_include "/usr/include/x86_64-linux-gnu" CACHE PATH "cuDNN include directory" FORCE)
	ENDIF ()
	IF (cudnn AND cudnn_include)
		MESSAGE (STATUS "Found cuDNN library: " ${cudnn})
		ADD_COMPILE_DEFINITIONS (CUDNN) # TODO this needs to be renamed
		ADD_COMPILE_DEFINITIONS (CUDNN_HALF)
		LIST (APPEND DARKNET_LINK_LIBS ${cudnn})
		MESSAGE (STATUS "Found cuDNN include: " ${cudnn_include})
		INCLUDE_DIRECTORIES (${cudnn_include})
		# -- cuDNN Frontend (FP8 graph convolution + FP4 block-scale) --
		SET (DARKNET_CUDNN_FRONTEND_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/cudnn-frontend/include")
		IF (EXISTS "${DARKNET_CUDNN_FRONTEND_INCLUDE_DIR}/cudnn_frontend.h")
			MESSAGE (STATUS "Found vendored cudnn-frontend include: ${DARKNET_CUDNN_FRONTEND_INCLUDE_DIR}")
			SET (DARKNET_FP8_CUDNN_CONV ${DARKNET_BUILD_FP8})
			INCLUDE_DIRECTORIES (${DARKNET_CUDNN_FRONTEND_INCLUDE_DIR})
			FOREACH (darknet_cudnn_extra_lib cudnn_graph cudnn_engines_runtime_compiled cudnn_engines_precompiled cudnn_heuristic)
				FIND_LIBRARY(${darknet_cudnn_extra_lib} ${darknet_cudnn_extra_lib}
							HINTS ${CUDAToolkit_LIBRARY_DIR} ${darknet_cuda_search_roots} ENV CUDNN_LIBRARY_DIR ENV CUDA_PATH ENV CUDNN_HOME
							PATHS /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu /usr/local /usr/local/cuda ENV LD_LIBRARY_PATH
							PATH_SUFFIXES lib64 lib/x64 lib x64 targets/x86_64-linux/lib targets/x86_64-linux/lib/stubs)
			ENDFOREACH ()
			IF ((DARKNET_BUILD_FP8 OR DARKNET_BUILD_FP4) AND cudnn_graph AND cudnn_engines_runtime_compiled)
				MESSAGE (STATUS "Found cuDNN graph library: ${cudnn_graph}")
				MESSAGE (STATUS "Found cuDNN runtime-compiled engines library: ${cudnn_engines_runtime_compiled}")
				IF (DARKNET_BUILD_FP8)
					ADD_COMPILE_DEFINITIONS (DARKNET_FP8_CUDNN_CONV)
				ENDIF ()
				LIST (APPEND DARKNET_LINK_LIBS ${cudnn_graph} ${cudnn_engines_runtime_compiled})
				IF (cudnn_engines_precompiled)
					LIST (APPEND DARKNET_LINK_LIBS ${cudnn_engines_precompiled})
				ENDIF ()
				IF (cudnn_heuristic)
					LIST (APPEND DARKNET_LINK_LIBS ${cudnn_heuristic})
				ENDIF ()

				# -- FP4 block-scale API detection --
				# FP4 uses cuDNN Frontend block-scale quantize/dequantize + matmul
				# (fe::graph::Block_scale_quantize_attributes /
				# Block_scale_dequantize_attributes, see src-fp4/fp4_gemm.cpp).
				# Rather than pin a specific cuDNN/frontend version number (which
				# has to be hand-updated to match whatever happens to be installed
				# on each machine), feature-detect the real API directly: if it
				# compiles against the headers actually found, FP4 is supported
				# here, regardless of the installed version.
				SET (DARKNET_FP4_CUDNN_924 OFF)
				IF (DARKNET_BUILD_FP4)
					INCLUDE (CheckCXXSourceCompiles)
					# CHECK_CXX_SOURCE_COMPILES compiles *and links* a full executable.
					# cudnn_frontend's Block_scale_quantize/dequantize path pulls in
					# non-header-only symbols (cudnnGetVersion, cudnnBackendCreateDescriptor,
					# etc.), so CMAKE_REQUIRED_LIBRARIES must include libcudnn or this check
					# fails to link on every machine regardless of whether the real API is
					# available -- silently disabling FP4 everywhere, including genuine
					# Blackwell hardware with a fully capable cuDNN install.
					SET (CMAKE_REQUIRED_INCLUDES ${cudnn_include} ${DARKNET_CUDNN_FRONTEND_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS})
					SET (CMAKE_REQUIRED_LIBRARIES ${cudnn})
					CHECK_CXX_SOURCE_COMPILES("#include <cudnn_frontend.h>\nint main() {\n  namespace fe = cudnn_frontend;\n  auto q = fe::graph::Block_scale_quantize_attributes().set_block_size(16).set_axis(2).set_transpose(false);\n  auto dq = fe::graph::Block_scale_dequantize_attributes().set_block_size({1,16}).set_is_negative_scale(false);\n  return 0;\n}" DARKNET_CUDNN_FP4_924_API)
					UNSET (CMAKE_REQUIRED_INCLUDES)
					UNSET (CMAKE_REQUIRED_LIBRARIES)

					IF (DARKNET_CUDNN_FP4_924_API AND cudnn_heuristic)
						SET (DARKNET_FP4_CUDNN_924 ON)
						ADD_COMPILE_DEFINITIONS (DARKNET_HAS_FP4_CUDNN_924)
						MESSAGE (STATUS "FP4 enabled: CUDA ${CUDAToolkit_VERSION}, cuDNN block-scale quantize/dequantize API found")
					ELSE ()
						MESSAGE (WARNING "FP4 disabled: requires compute capability >= 100 (Blackwell) and CUDA >= 13.2")
						darknet_disable_fp4 ("FP4 disabled: requires compute capability >= 100 (Blackwell) and CUDA >= 13.2")
					ENDIF ()
				ENDIF ()
			ELSE ()
				MESSAGE (WARNING "cuDNN graph runtime libraries not found; FP8 cuDNN graph convolution disabled.")
				SET (DARKNET_FP8_CUDNN_CONV OFF)
				darknet_disable_fp4 ("DARKNET_ENABLE_FP4=ON requires cuDNN Frontend and cuDNN graph runtime libraries")
			ENDIF ()
		ELSE ()
			MESSAGE (STATUS "Vendored cudnn-frontend not found; FP8 cuDNN graph convolution disabled.")
			darknet_disable_fp4 ("DARKNET_ENABLE_FP4=ON requires the vendored cuDNN Frontend headers")
		ENDIF ()
	ELSE ()
		MESSAGE (WARNING "cuDNN not found.")
		darknet_disable_fp4 ("DARKNET_ENABLE_FP4=ON requires cuDNN")
	ENDIF ()
ENDIF ()


# ======================
# == AMD GPU aka ROCM ==
# ======================
CMAKE_DEPENDENT_OPTION (DARKNET_TRY_ROCM "Attempt to find AMD/ROCm/HIP GPU support" ON "" ON)
IF (DARKNET_TRY_ROCM)
	CHECK_LANGUAGE (HIP)
	IF (CMAKE_HIP_COMPILER)
		MESSAGE (STATUS "AMD ROCm detected. Darknet will use AMD GPUs. HIP compiler is ${CMAKE_HIP_COMPILER}.")
		IF (NOT DEFINED ROCM_PATH)
			SET (ROCM_PATH "/opt/rocm")
		ENDIF ()
		LIST (APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})
		ENABLE_LANGUAGE (HIP)
		FIND_PACKAGE(hip REQUIRED)
		FIND_PACKAGE(hipblas REQUIRED)
		FIND_PACKAGE(hiprand REQUIRED)
		FIND_PACKAGE(amd_smi REQUIRED)

		SET (DARKNET_USE_ROCM ON)

		SET (CMAKE_HIP_STANDARD 17)
		SET (CMAKE_HIP_STANDARD_REQUIRED ON)

		SET (CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
		SET (CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})

		ADD_COMPILE_DEFINITIONS (__HIP_PLATFORM_HCC__)
		ADD_COMPILE_DEFINITIONS (__HIP_PLATFORM_AMD__)
		ADD_COMPILE_DEFINITIONS (DARKNET_GPU_ROCM)
		ADD_COMPILE_DEFINITIONS (DARKNET_GPU)

		INCLUDE_DIRECTORIES ("${ROCM_PATH}/include/")

		# Run "rocm-smi --showproductname" or "rocm-smi --showhw" to see which architecture to use.
		# For example, this can be set to "gfx1035;gfx1036;gfx1037" to build code for multiple architectures.
		#
		#	gfx1101: RX 7700 / 7800
		#
		IF (NOT DEFINED CMAKE_HIP_ARCHITECTURES)
			SET (CMAKE_HIP_ARCHITECTURES "gfx1101")
		ENDIF ()

		LIST (APPEND DARKNET_LINK_LIBS hip::host hip::device roc::hipblas roc::rocrand hip::hiprand amd_smi)

#		MESSAGE (STATUS "Enabling hipDNN")
#		ADD_COMPILE_DEFINITIONS (CUDNN) # TODO this needs to be renamed
#		ADD_COMPILE_DEFINITIONS (CUDNN_HALF)

	ELSE ()
		MESSAGE (WARNING "Support for AMD/ROCm/HIP not found.")
	ENDIF ()
ELSE ()
	MESSAGE (WARNING "Support for AMD/ROCm/HIP is disabled.")
ENDIF ()


# =====================
# == Apple Metal/MPS ==
# =====================
IF (APPLE)
	CMAKE_DEPENDENT_OPTION (DARKNET_TRY_MPS "Attempt to find Apple Metal/MPS support" ON "" ON)
	IF (DARKNET_TRY_MPS)
		FIND_LIBRARY (APPLE_METAL Metal)
		FIND_LIBRARY (APPLE_MPS MetalPerformanceShaders)
		FIND_LIBRARY (APPLE_FOUNDATION Foundation)
		IF (APPLE_METAL AND APPLE_MPS AND APPLE_FOUNDATION)
			MESSAGE (STATUS "Apple Metal/MPS detected. Darknet will use MPS for inference acceleration.")
			SET (DARKNET_USE_MPS ON)
			SET (CMAKE_OBJCXX_STANDARD 23)
			SET (CMAKE_OBJCXX_STANDARD_REQUIRED ON)
			ENABLE_LANGUAGE (OBJCXX)
			ADD_COMPILE_DEFINITIONS (DARKNET_USE_MPS)
			LIST (APPEND DARKNET_LINK_LIBS ${APPLE_METAL} ${APPLE_MPS} ${APPLE_FOUNDATION})
		ELSE ()
			MESSAGE (WARNING "Apple Metal/MPS not found.")
		ENDIF ()
	ELSE ()
		MESSAGE (WARNING "Apple Metal/MPS support is disabled.")
	ENDIF ()
ENDIF ()


# ==============
# == CPU-only ==
# ==============
IF (NOT DARKNET_USE_CUDA AND NOT DARKNET_USE_ROCM AND NOT DARKNET_USE_MPS)
	SET (DARKNET_DETECTED_CPU_ONLY TRUE)
	MESSAGE (WARNING "Neither NVIDIA CUDA nor AMD ROCm detected.  Darknet will be CPU-only.")
ENDIF ()


# ========================
# == Intel/AMD Hardware ==
# ========================
IF (CMAKE_SYSTEM_PROCESSOR MATCHES "x86" OR
	CMAKE_SYSTEM_PROCESSOR MATCHES "x86_32" OR
	CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64" OR
	CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64")
	SET (HARDWARE_IS_X86 TRUE)
	MESSAGE (STATUS "Hardware is 32-bit or 64-bit, and seems to be Intel or AMD:  ${CMAKE_SYSTEM_PROCESSOR}")
ELSE ()
	SET (HARDWARE_IS_X86 FALSE)
	MESSAGE (STATUS "Hardware does not appear to be 32-bit or 64-bit, Intel or AMD:  ${CMAKE_SYSTEM_PROCESSOR}")
ENDIF ()


# ===============
# == GCC/Clang ==
# ===============
IF (CMAKE_COMPILER_IS_GNUCC OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
	SET (COMPILER_IS_GNU_OR_CLANG TRUE)
ELSE ()
	SET (COMPILER_IS_GNU_OR_CLANG FALSE)
ENDIF ()


# ====================
# == GCC/Clang/MSCV ==
# ====================
IF (COMPILER_IS_GNU_OR_CLANG OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC")
	SET (COMPILER_IS_GNU_OR_CLANG_OR_MSVC TRUE)
ELSE ()
	SET (COMPILER_IS_GNU_OR_CLANG_OR_MSVC FALSE)
ENDIF ()

MESSAGE (STATUS "Compiler:  GNU/Clang=${CMAKE_COMPILER_IS_GNUCC} GNU/Clang/MSVC=${COMPILER_IS_GNU_OR_CLANG_OR_MSVC}: ${CMAKE_CXX_COMPILER_ID}")


# =============
# == Threads ==
# =============
FIND_PACKAGE (Threads REQUIRED)
MESSAGE (STATUS "Found Threads ${Threads_VERSION}")
LIST (APPEND DARKNET_LINK_LIBS Threads::Threads)

# ============================================================
# == OpenBLAS (Basic Linear Algebra Subprograms)			==
# == This is only used when Darknet is built for CPU-only.	==
# ============================================================
IF (DARKNET_DETECTED_CPU_ONLY)

	IF (NOT DEFINED DARKNET_TRY_OPENBLAS)
		SET (DARKNET_TRY_OPENBLAS True)
	ENDIF ()

	IF (DARKNET_TRY_OPENBLAS)
		SET(BLA_VENDOR OpenBLAS)
		SET(BLA_SIZEOF_INTEGER 8) # force 64 bit
		IF (APPLE)
			# APPLE devices need a hint to find the brew installation.  On top of which, on some distrios (and again APPLE)
			# the package is called OpenBLAS, while on other distros it is called OpenBLAS64.  We need to search for both.
			FIND_PACKAGE (OpenBLAS NAMES OpenBLAS64 OpenBLAS QUIET HINTS "/opt/homebrew/opt/openblas/lib/cmake/openblas")
			IF (OpenBLAS_FOUND)
				LIST (APPEND DARKNET_LINK_LIBS ${OpenBLAS_LIBRARIES})
				INCLUDE_DIRECTORIES (${OpenBLAS_INCLUDE_DIRS})
				ADD_COMPILE_DEFINITIONS (DARKNET_USE_OPENBLAS)
			ELSE ()
				MESSAGE (WARNING "Apple OpenBLAS not found. Building Darknet for CPU-only without support for OpenBLAS.")
			ENDIF ()
		ELSEIF (WIN32)
			FIND_PACKAGE (OpenBLAS)
			IF (OpenBLAS_FOUND)
				MESSAGE (STATUS "Found OpenBLAS")
				LIST (APPEND DARKNET_LINK_LIBS OpenBLAS::OpenBLAS)
				ADD_COMPILE_DEFINITIONS (DARKNET_USE_OPENBLAS)
			ELSE ()
				MESSAGE (WARNING "OpenBLAS not found. Building Darknet for CPU-only without support for OpenBLAS.")
			ENDIF ()
		ELSE ()
			FIND_PACKAGE (BLAS)
			IF (BLAS_FOUND)
				MESSAGE (STATUS "Found OpenBLAS")
				LIST (APPEND DARKNET_LINK_LIBS BLAS::BLAS)
				ADD_COMPILE_DEFINITIONS (DARKNET_USE_OPENBLAS)
			ELSE ()
				MESSAGE (WARNING "OpenBLAS not found. Building Darknet for CPU-only without support for OpenBLAS.")
			ENDIF ()
		ENDIF ()
	ELSE ()
		MESSAGE (WARNING "OpenBLAS is disabled. Building Darknet for CPU-only without support for OpenBLAS.")
	ENDIF ()
ELSE ()
	MESSAGE (STATUS "Skipping OpenBLAS since we have a GPU.")
ENDIF ()

# ============
# == OpenCV ==
# ============
FIND_PACKAGE (OpenCV REQUIRED)
MESSAGE (STATUS "Found OpenCV ${OpenCV_VERSION}")
INCLUDE_DIRECTORIES (${OpenCV_INCLUDE_DIRS})
LIST (APPEND DARKNET_LINK_LIBS ${OpenCV_LIBS})

# ============
# == OpenMP ==
# ============
FIND_PACKAGE (OpenMP QUIET) # optional
IF (NOT OPENMP_FOUND)
	MESSAGE (WARNING "OpenMP not found. Building Darknet without support for OpenMP.")
ELSEIF (DARKNET_USE_ROCM)
	# TODO: This needs to be fixed.  What are we missing during the link process to make this work with clang++?
	MESSAGE (WARNING "Skipping OpenMP due to ROCm.")
ELSE ()
	MESSAGE (STATUS "Found OpenMP ${OpenMP_VERSION}")
	ADD_COMPILE_DEFINITIONS (DARKNET_OPENMP)
	LIST (APPEND DARKNET_LINK_LIBS OpenMP::OpenMP_CXX OpenMP::OpenMP_C)
	IF (WIN32)
		ADD_COMPILE_OPTIONS (/openmp:experimental)
	ELSE ()
		ADD_COMPILE_DEFINITIONS (_GLIBCXX_PARALLEL)
		ADD_COMPILE_OPTIONS (-fopenmp)
		ADD_COMPILE_OPTIONS (${OpenMP_C_FLAGS})
		ADD_COMPILE_OPTIONS (${OpenMP_CXX_FLAGS})
	ENDIF()
ENDIF ()


# ===============
# == AVX & SSE ==
# ===============
CMAKE_DEPENDENT_OPTION (ENABLE_SSE_AND_AVX "Enable AVX and SSE optimizations (Intel and AMD only)" ON "COMPILER_IS_GNU_OR_CLANG_OR_MSVC;HARDWARE_IS_X86" OFF)
IF (NOT ENABLE_SSE_AND_AVX)
	MESSAGE (WARNING "AVX and SSE optimizations are disabled.")
ELSE ()
	MESSAGE (STATUS "Enabling AVX and SSE optimizations.")
	IF (COMPILER_IS_GNU_OR_CLANG)
		ADD_COMPILE_OPTIONS(-ffp-contract=fast)
		ADD_COMPILE_OPTIONS(-mavx)
		ADD_COMPILE_OPTIONS(-mavx2)
		ADD_COMPILE_OPTIONS(-msse3)
		ADD_COMPILE_OPTIONS(-msse4.1)
		ADD_COMPILE_OPTIONS(-msse4.2)
		ADD_COMPILE_OPTIONS(-msse4a)
	ELSE ()
		STRING (APPEND CMAKE_CXX_FLAGS " /arch:AVX2")
	ENDIF()
ENDIF ()


# ============
# == Timing ==
# ============
CMAKE_DEPENDENT_OPTION (ENABLE_TIMING_AND_TRACKING "Enable Darknet timing and tracking debugging" OFF "" OFF)
IF (ENABLE_TIMING_AND_TRACKING)
	MESSAGE (WARNING "Darknet timing and tracking debug code is *ENABLED*!")
	ADD_COMPILE_DEFINITIONS(DARKNET_TIMING_AND_TRACKING_ENABLED)
ENDIF ()


# ===================================
# == Protocol Buffer (ONNX export) ==
# ===================================
IF (NOT DEFINED DARKNET_TRY_ONNX)
	SET (DARKNET_TRY_ONNX True)
ENDIF ()
IF (DARKNET_TRY_ONNX)
	FIND_PACKAGE (Protobuf QUIET)
	IF (Protobuf_FOUND)
		MESSAGE (STATUS "Found protocol buffer (needed for ONNX export) ${Protobuf_VERSION}")
		INCLUDE_DIRECTORIES (${Protobuf_INCLUDE_DIRS})
		ADD_COMPILE_DEFINITIONS(DARKNET_HAS_PROTOBUF)
	ELSE ()
		MESSAGE (WARNING "Protocol buffer not found.  Skipping support for ONNX export.")
	ENDIF ()
ELSE ()
	MESSAGE (STATUS "Darknet is skipping ONNX.  Run cmake with '-DDARKNET_TRY_ONNX=True' to add support for the ONNX export tool.")
ENDIF ()
