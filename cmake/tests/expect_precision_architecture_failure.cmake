cmake_minimum_required(VERSION 3.24)
include("${CMAKE_CURRENT_LIST_DIR}/../DarknetPrecisionArchitectures.cmake")
darknet_resolve_precision_architectures(
	"${ARCHITECTURES}" "89"
	has_fp8 has_fp4 has_sm89 resolved fp8_architectures fp4_architectures)
