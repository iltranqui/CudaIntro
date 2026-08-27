# Darknet object detection framework


# ==
# Darknet CLI is actually quite simple in terms of source files.
# ==
MESSAGE(STATUS "Setting up DARKNET CLI")

FILE (GLOB CLISRC *.cpp)
IF (WIN32)
	INCLUDE_DIRECTORIES (windows)
	FILE (GLOB WINRC windows/*.rc)
	LIST (APPEND CLISRC ${WINRC})
ENDIF ()

LIST (SORT CLISRC)

SET (DARKNET_CLI_PRECISION_OBJECTS)
IF (DARKNET_BUILD_FP8)
	LIST (APPEND DARKNET_CLI_PRECISION_OBJECTS $<TARGET_OBJECTS:darknet_fp8_objlib>)
ENDIF ()
IF (DARKNET_BUILD_FP4)
	LIST (APPEND DARKNET_CLI_PRECISION_OBJECTS $<TARGET_OBJECTS:darknet_fp4_objlib>)
ENDIF ()

# darknetcli links darknetobjlib directly rather than the shared library, so it
# must also consume the optional precision object files.  Without this, a CUDA
# build can configure FP4 yet produce a CLI with unresolved FP4 symbols.
ADD_EXECUTABLE (darknetcli ${CLISRC} $<TARGET_OBJECTS:darknetobjlib> ${DARKNET_CLI_PRECISION_OBJECTS})
SET_TARGET_PROPERTIES (darknetcli PROPERTIES OUTPUT_NAME "darknet")
IF (DARKNET_BUILD_FP8)
	TARGET_COMPILE_DEFINITIONS (darknetcli PRIVATE DARKNET_HAS_FP8)
ENDIF ()
IF (DARKNET_BUILD_FP4)
	# The precision object libraries carry this definition themselves, but the
	# CLI benchmark also needs it to expose its execution-proof counters.
	TARGET_COMPILE_DEFINITIONS (darknetcli PRIVATE DARKNET_HAS_FP4)
ENDIF ()
IF (DARKNET_USE_CUDA OR DARKNET_USE_ROCM)
	SET_TARGET_PROPERTIES (darknetcli PROPERTIES CUDA_ARCHITECTURES "${DARKNET_CUDA_ARCHITECTURES}")
	SET_TARGET_PROPERTIES (darknetcli PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
	SET_TARGET_PROPERTIES (darknetcli PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
ENDIF ()
TARGET_LINK_LIBRARIES (darknetcli PRIVATE ${DARKNET_LINK_LIBS})

IF (UNIX)
	# non-Windows installation is very simple
	INSTALL (TARGETS darknetcli DESTINATION bin)
ELSE ()
	# more complicated install for Windows so we also get the .DLL files copied over from vcpkg
	# https://stackoverflow.com/a/72088102/13022
	INSTALL (TARGETS darknetcli
		DESTINATION bin
		RUNTIME_DEPENDENCIES
			PRE_EXCLUDE_REGEXES "api-ms-" "ext-ms-" "wpaxholder" "HvsiFileTrust" "PdmUtilities"
			POST_EXCLUDE_REGEXES ".*system32/.*\\.dll"
			DIRECTORIES ${CMAKE_BINARY_DIR}/bin
		RUNTIME DESTINATION bin
		)
ENDIF ()
