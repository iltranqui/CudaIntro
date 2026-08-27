# Darknet object detection framework


MESSAGE(STATUS "Setting up DARKNET unit tests")

INCLUDE_DIRECTORIES (/usr/src/googletest/googletest/include/)

SET (TEST_SOURCE
	test_blas_kernels.cpp
	test_box.cpp
	test_image.cpp
	test_main.cpp
	test_misc.cpp
	test_random.cpp
	test_clifford_layer.cpp
	test_deform_conv.cpp
	test_dcnv4.cpp
	test_graph_conv.cpp
	test_custom_layers.cpp
	test_conv_precision_route.cpp
	test_fp4_convolution_lowering.cpp
	test_tucker_attention.cpp
	test_vit_layer.cpp
	test_modern_yolo_heads.cpp
	test_yolo_layer_cuda.cpp
	test_time.cpp
)

LIST (APPEND TEST_SOURCE test_fp4_scaling.cpp)
IF (NOT DARKNET_BUILD_FP4)
	LIST (APPEND TEST_SOURCE ../src-fp4/fp4_scaling.cpp)
ENDIF ()

IF (DARKNET_BUILD_FP8)
	LIST (APPEND TEST_SOURCE
		test_fp8_calibration.cpp
		test_fp8_conv.cpp
		test_fp8_gemm.cpp
		test_fp8_kernels.cpp
		test_fp8_scaling.cpp)
ENDIF ()

ADD_EXECUTABLE (darknet_tests ${TEST_SOURCE})
TARGET_INCLUDE_DIRECTORIES (darknet_tests PRIVATE ../src-fp4)
TARGET_LINK_LIBRARIES ( darknet_tests PRIVATE darknet ${DARKNET_LINK_LIBS} ${GTEST})

ADD_TEST (NAME darknet_tests COMMAND darknet_tests)

IF (Protobuf_FOUND)
	SET (FP4_ONNX_STATE_TEST_SOURCE
		test_main.cpp
		test_fp4_onnx_state_isolation.cpp
		../src-onnx/darknet_onnx.cpp
		../src-onnx/darknet_node.cpp
		../src-onnx/onnx.proto3.pb.cc)
	IF (NOT DARKNET_BUILD_FP4)
		LIST (APPEND FP4_ONNX_STATE_TEST_SOURCE ../src-fp4/fp4_scaling.cpp)
	ENDIF ()
	ADD_EXECUTABLE (darknet_fp4_onnx_state_tests
		${FP4_ONNX_STATE_TEST_SOURCE})
	ADD_DEPENDENCIES (darknet_fp4_onnx_state_tests darknet_generated_proto)
	TARGET_INCLUDE_DIRECTORIES (darknet_fp4_onnx_state_tests PRIVATE ../src-fp4 ../src-onnx)
	TARGET_LINK_LIBRARIES (darknet_fp4_onnx_state_tests PRIVATE darknet ${DARKNET_LINK_LIBS} ${GTEST} protobuf::libprotobuf)
	ADD_TEST (NAME darknet_fp4_onnx_state_tests COMMAND darknet_fp4_onnx_state_tests)
ENDIF ()

# Storage conversion is intentionally testable on Ada: it does not execute an
# FP4 Tensor Core instruction.  Keep this target separate from the Blackwell-
# gated production object library so an SM89 developer machine can verify it.
IF (DARKNET_USE_CUDA AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.8)
	SET (FP4_KERNEL_TEST_SOURCE test_main.cpp test_fp4_kernels.cpp ../src-fp4/fp4_kernels.cu)
	IF (NOT DARKNET_BUILD_FP4)
		LIST (APPEND FP4_KERNEL_TEST_SOURCE ../src-fp4/fp4_scaling.cpp)
	ENDIF ()
	ADD_EXECUTABLE (fp4_kernel_tests ${FP4_KERNEL_TEST_SOURCE})
	TARGET_INCLUDE_DIRECTORIES (fp4_kernel_tests PRIVATE ../src-fp4 ../src-lib)
	TARGET_LINK_LIBRARIES (fp4_kernel_tests PRIVATE darknet ${DARKNET_LINK_LIBS} ${GTEST})
	SET_TARGET_PROPERTIES (fp4_kernel_tests PROPERTIES
		CUDA_ARCHITECTURES "${DARKNET_CUDA_ARCHITECTURES}"
		CUDA_SEPARABLE_COMPILATION OFF
		CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
	ADD_TEST (NAME fp4_kernel_tests COMMAND fp4_kernel_tests)
ENDIF ()
