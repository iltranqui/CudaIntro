# Darknet object detection framework


# Create a version string from the git tag and commit hash (see src/darknet_version.h.in).
# Should look similar to this:
#
#		v1.99-63-gc5c3569
#
EXECUTE_PROCESS (COMMAND git describe --tags --dirty --long --always OUTPUT_VARIABLE DARKNET_VERSION_STRING OUTPUT_STRIP_TRAILING_WHITESPACE)

STRING (REGEX MATCH "v([0-9]+)\.([0-9]+)-([0-9]+)-g([0-9a-fA-F]+)" DARKNET_VERSION_MATCH ${DARKNET_VERSION_STRING})
# note that MATCH_4 is not numeric

IF (DARKNET_VERSION_MATCH STREQUAL "")
	# no reachable tag -- git describe --always fell back to a bare commit hash instead of
	# "vX.Y-N-gHASH".  This happens on v6-dev-derived branches (e.g. fp8) that were rebased /
	# cherry-picked rather than merged, so they carry their own copy of the "new branch for v6
	# development" commit (905ccee6) instead of being a descendant of the v6.0 tag.  Count
	# commits since that marker so the version still increases by one on every commit.
	EXECUTE_PROCESS (COMMAND git rev-list --count 905ccee6..HEAD OUTPUT_VARIABLE DARKNET_VERSION_COMMIT_COUNT OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET RESULT_VARIABLE DARKNET_VERSION_COMMIT_COUNT_RESULT)
	IF (DARKNET_VERSION_COMMIT_COUNT_RESULT EQUAL 0)
		SET (DARKNET_VERSION_SHORT 6.0.${DARKNET_VERSION_COMMIT_COUNT})
	ELSE ()
		# 905ccee6 not present in this clone's history -- fall back to a placeholder version
		SET (DARKNET_VERSION_SHORT "0.0.0")
	ENDIF ()
	# rebuild the display string with the computed version in front of the bare hash/dirty
	# suffix "git describe --always" gave us, e.g. "6.0.176-dcbd0a2a-dirty"
	SET (DARKNET_VERSION_STRING "${DARKNET_VERSION_SHORT}-${DARKNET_VERSION_STRING}")
ELSE ()
	SET (DARKNET_VERSION_SHORT ${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3})
ENDIF ()

EXECUTE_PROCESS (COMMAND git branch --show-current OUTPUT_VARIABLE DARKNET_BRANCH_NAME OUTPUT_STRIP_TRAILING_WHITESPACE)
IF (DARKNET_BRANCH_NAME STREQUAL "")
	SET (DARKNET_BRANCH_NAME "unknown")
ENDIF ()

# Mirror the "[branch]" suffix that Darknet::show_version_info() prints at runtime, so the
# same version+branch banner is visible in the CMake configure log, not just "darknet --version".
IF (DARKNET_BRANCH_NAME STREQUAL "master")
	MESSAGE (STATUS "Darknet ${DARKNET_VERSION_STRING}")
ELSE ()
	MESSAGE (STATUS "Darknet ${DARKNET_VERSION_STRING} [${DARKNET_BRANCH_NAME}]")
ENDIF ()
MESSAGE (STATUS "Darknet branch name: ${DARKNET_BRANCH_NAME}")
