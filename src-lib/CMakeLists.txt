# Darknet object detection framework


SET (CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON) # automatically create a module definition file with all global symbols
CONFIGURE_FILE (${CMAKE_CURRENT_SOURCE_DIR}/darknet_version.h.in ${CMAKE_CURRENT_SOURCE_DIR}/darknet_version.h)


# ==
# The library is somewhat more complicated than the original CLI.  It has lots of source files, more optional files
# depending on whether or not CUDA is enabled, and yet more files if we're building on Windows.
# ==
FILE (GLOB LIBSRC *.cpp)
LIST (REMOVE_ITEM LIBSRC
	yolo_v2_class.cpp
	)
LIST (FILTER LIBSRC EXCLUDE REGEX ".*/patch_.*\\.cpp$")
LIST (FILTER LIBSRC EXCLUDE REGEX ".*/fp8_conv_probe\\.cpp$")

IF (DARKNET_USE_MPS)
	INCLUDE_DIRECTORIES (apple)
	LIST (APPEND LIBSRC apple/apple_mps.mm)
	LIST (APPEND LIBSRC apple/metal_backend.mm)
	SET_SOURCE_FILES_PROPERTIES (apple/apple_mps.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
	SET_SOURCE_FILES_PROPERTIES (apple/metal_backend.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
ENDIF ()

IF (COMPILER_IS_GNU_OR_CLANG)
	IF (CMAKE_BUILD_TYPE MATCHES RELEASE OR
		CMAKE_BUILD_TYPE MATCHES Release OR
		CMAKE_BUILD_TYPE MATCHES release)
		# nvcc doesn't like this flag, so we have to set it specifically on the .cpp files and not on the .cu files
		SET_SOURCE_FILES_PROPERTIES (${LIBSRC} PROPERTIES COMPILE_FLAGS -Ofast)
	ENDIF ()
ENDIF ()

IF (DARKNET_USE_CUDA OR DARKNET_USE_ROCM)
	MESSAGE (STATUS "Adding .cu files for GPU build...")
	FILE (GLOB CUDASRC *.cu)
	LIST (APPEND LIBSRC ${CUDASRC})

	IF (DARKNET_USE_ROCM)
		SET_SOURCE_FILES_PROPERTIES (${CUDASRC} PROPERTIES LANGUAGE HIP)
	ENDIF ()

ENDIF ()

IF (WIN32)
	INCLUDE_DIRECTORIES (windows)
	FILE (GLOB WINSRC windows/*.c)
	LIST (APPEND LIBSRC ${WINSRC})
ENDIF ()

LIST (SORT LIBSRC)


# ==
# First thing we do is build an object library so things don't have to be compiled multiple times.
# ==
MESSAGE(STATUS "Setting up DARKNET OBJ")
ADD_LIBRARY(darknetobjlib OBJECT ${LIBSRC})
IF (Protobuf_FOUND)
	# force the protocol buffer compiler to run if it hasn't yet
	ADD_DEPENDENCIES (darknetobjlib darknet_generated_proto)
ENDIF ()
IF (DARKNET_USE_CUDA OR DARKNET_USE_ROCM)
	SET_TARGET_PROPERTIES (darknetobjlib PROPERTIES CUDA_ARCHITECTURES "${DARKNET_CUDA_ARCHITECTURES}")
	SET_TARGET_PROPERTIES (darknetobjlib PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
	SET_TARGET_PROPERTIES (darknetobjlib PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
ENDIF ()
TARGET_LINK_LIBRARIES (darknetobjlib PRIVATE ${DARKNET_LINK_LIBS})

set(DARKNET_PRECISION_OBJECTS "")
if(DARKNET_BUILD_FP8)
	message(STATUS "DARKNET_FP8_SOURCES enabled")
	add_library(darknet_fp8_objlib OBJECT
		../src-fp8/fp8_calibration.cpp
		../src-fp8/fp8_conv.cpp
		../src-fp8/fp8_gemm.cpp
		../src-fp8/fp8_scaling.cpp
		../src-fp8/fp8_kernels.cu
		../src-fp8/fp8_attention_kernels.cu)
	target_compile_definitions(darknetobjlib PRIVATE DARKNET_HAS_FP8)
	target_compile_definitions(darknet_fp8_objlib PRIVATE DARKNET_HAS_FP8)
	if(DARKNET_FP8_TARGET_SM89)
		target_compile_definitions(darknetobjlib PRIVATE DARKNET_FP8_TARGET_SM89)
		target_compile_definitions(darknet_fp8_objlib PRIVATE DARKNET_FP8_TARGET_SM89)
	endif()
	target_include_directories(darknet_fp8_objlib PRIVATE . ../src-fp8 ${CUDAToolkit_INCLUDE_DIRS})
	target_link_libraries(darknet_fp8_objlib PRIVATE ${DARKNET_LINK_LIBS})
	set_target_properties(darknet_fp8_objlib PROPERTIES
		CUDA_ARCHITECTURES "${DARKNET_FP8_CUDA_ARCHITECTURES}"
		CUDA_SEPARABLE_COMPILATION OFF
		CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
	list(APPEND DARKNET_PRECISION_OBJECTS $<TARGET_OBJECTS:darknet_fp8_objlib>)
else()
	message(STATUS "DARKNET_FP8_SOURCES disabled")
endif()

if(DARKNET_BUILD_FP4)
	message(STATUS "DARKNET_FP4_SOURCES enabled")
	add_library(darknet_fp4_objlib OBJECT
		../src-fp4/fp4_attention_kernels.cu
		../src-fp4/fp4_calibration.cpp
		../src-fp4/fp4_gemm.cpp
		../src-fp4/fp4_kernels.cu
		../src-fp4/fp4_scaling.cpp)
	target_compile_definitions(darknetobjlib PRIVATE DARKNET_HAS_FP4)
	target_compile_definitions(darknet_fp4_objlib PRIVATE DARKNET_HAS_FP4)
	target_include_directories(darknet_fp4_objlib PRIVATE . ../src-fp4 ${DARKNET_CUDNN_FRONTEND_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS})
	target_link_libraries(darknet_fp4_objlib PRIVATE ${DARKNET_LINK_LIBS})
	set_target_properties(darknet_fp4_objlib PROPERTIES CUDA_ARCHITECTURES "${DARKNET_FP4_CUDA_ARCHITECTURES}")
	list(APPEND DARKNET_PRECISION_OBJECTS $<TARGET_OBJECTS:darknet_fp4_objlib>)
else()
	message(STATUS "DARKNET_FP4_SOURCES disabled")
endif()


# ==
# Next step is to create the actual darknet library.
# ==
MESSAGE(STATUS "Setting up DARKNET LIB")
SET (DARKNET_PUBLIC_HEADERS
	darknet.h
	darknet.hpp
	darknet_args_and_parms.hpp
	darknet_cfg_and_state.hpp
	darknet_cfg.hpp
	darknet_image.hpp
	darknet_keypoints.hpp
	darknet_version.h
	)
ADD_LIBRARY (darknet SHARED $<TARGET_OBJECTS:darknetobjlib> ${DARKNET_PRECISION_OBJECTS})
SET_TARGET_PROPERTIES (darknet PROPERTIES PUBLIC_HEADER "${DARKNET_PUBLIC_HEADERS}")
IF (DARKNET_USE_CUDA OR DARKNET_USE_ROCM)
	SET_TARGET_PROPERTIES (darknet PROPERTIES CUDA_ARCHITECTURES "${DARKNET_CUDA_ARCHITECTURES}")
	SET_TARGET_PROPERTIES (darknet PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
	SET_TARGET_PROPERTIES (darknet PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
ENDIF ()
TARGET_LINK_LIBRARIES (darknet PRIVATE ${DARKNET_LINK_LIBS})
INSTALL (TARGETS darknet LIBRARY DESTINATION lib PUBLIC_HEADER DESTINATION include)

IF (DARKNET_FP8_CUDNN_CONV)
	ADD_EXECUTABLE (fp8_conv_probe ../src-fp8/fp8_conv_probe.cpp)
	TARGET_LINK_LIBRARIES (fp8_conv_probe PRIVATE ${DARKNET_LINK_LIBS})
	TARGET_INCLUDE_DIRECTORIES (fp8_conv_probe PRIVATE
		${cudnn_include}
		${DARKNET_CUDNN_FRONTEND_INCLUDE_DIR}
		${CUDAToolkit_INCLUDE_DIRS}
	)
ENDIF ()


# ==
# The old C++ YOLOv2 library is still used by some projects.  E.g., the original C# wrapper.
# But as far as I can tell, this library was only ever used on Windows.  Doesn't look like
# the old Linux Makefile ever tried to build this on Linux.
# ==
IF (WIN32)
	MESSAGE(STATUS "Setting up YOLO V2 LIB")
	ADD_LIBRARY (yolo_v2_class SHARED yolo_v2_class.cpp)
	SET_TARGET_PROPERTIES (yolo_v2_class PROPERTIES PUBLIC_HEADER "yolo_v2_class.hpp")
	TARGET_LINK_LIBRARIES (yolo_v2_class PRIVATE ${DARKNET_LINK_LIBS} darknet)
	INSTALL (TARGETS yolo_v2_class LIBRARY DESTINATION lib PUBLIC_HEADER DESTINATION include)
ENDIF ()
