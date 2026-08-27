cmake_minimum_required(VERSION 3.24)
include("${CMAKE_CURRENT_LIST_DIR}/../DarknetPrecisionArchitectures.cmake")

function(expect_architectures input native expected_fp8 expected_fp4 expected_sm89 expected_resolved expected_fp8_architectures expected_fp4_architectures)
	darknet_resolve_precision_architectures("${input}" "${native}" actual_fp8 actual_fp4 actual_sm89 actual_resolved actual_fp8_architectures actual_fp4_architectures)
	foreach(name fp8 fp4 sm89 resolved fp8_architectures fp4_architectures)
		if(NOT "${actual_${name}}" STREQUAL "${expected_${name}}")
			message(FATAL_ERROR "${input}: expected ${name}=${expected_${name}}, got ${actual_${name}}")
		endif()
	endforeach()
endfunction()

expect_architectures("86-real" "" OFF OFF OFF "86-real" "" "")
expect_architectures("native" "89" ON OFF ON "89" "89" "")
expect_architectures("native" "86;89;100" ON ON ON "86;89;100" "89;100" "100")
expect_architectures("86-real;89-real;100-real;120-real" "" ON ON ON "86-real;89-real;100-real;120-real" "89-real;100-real;120-real" "100-real;120-real")
expect_architectures("90-real" "" ON OFF OFF "90-real" "90-real" "")
expect_architectures("103;121-virtual" "" ON ON OFF "103;121-virtual" "103;121-virtual" "103;121-virtual")

foreach(invalid_architecture "" "86junk" "89-real-extra" "native-real" "sm_89" "89.0")
	execute_process(
		COMMAND "${CMAKE_COMMAND}"
			-DARCHITECTURES=${invalid_architecture}
			-P "${CMAKE_CURRENT_LIST_DIR}/expect_precision_architecture_failure.cmake"
		RESULT_VARIABLE invalid_result
		OUTPUT_QUIET
		ERROR_QUIET)
	if(invalid_result EQUAL 0)
		message(FATAL_ERROR "Malformed CUDA architecture '${invalid_architecture}' was accepted")
	endif()
endforeach()

darknet_resolve_precision_feature(AUTO OFF DARKNET_ENABLE_FP4 feature_auto_off)
darknet_resolve_precision_feature(AUTO ON DARKNET_ENABLE_FP4 feature_auto_on)
darknet_resolve_precision_feature(OFF ON DARKNET_ENABLE_FP4 feature_forced_off)
if(feature_auto_off OR NOT feature_auto_on OR feature_forced_off)
	message(FATAL_ERROR "tri-state feature resolution contract failed")
endif()

message(STATUS "Darknet precision architecture contracts passed")
