#!/usr/bin/env bash

# set TEST_ROOT:
TEST_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# set PROJECT_ROOT and make sure that is our working directory
PROJECT_ROOT="$(readlink -f "${TEST_ROOT}/..")"
cd "${PROJECT_ROOT}"

# set SCRIPT_ROOT
SCRIPT_ROOT="$(readlink -f "${PROJECT_ROOT}/scripts")"

. "${SCRIPT_ROOT}/.include.sh"
. "${TEST_ROOT}/test-include.sh"

apply_config_files

rm -rf runs/default/unnamed

echo "Cleaning up for '${TARGET_SQSH}' and '${TARGET_TAG}' (ENROOT_IMAGE_HOME='${ENROOT_IMAGE_HOME}')..."

rm -f "${ENROOT_IMAGE_HOME}/${TARGET_SQSH}"
NUM_SQSH="$( ls ${ENROOT_IMAGE_HOME} | grep "${TARGET_SQSH}" | wc -l )"
assert_equals "${NUM_SQSH}" "0"

NEXT_CONTAINER="$( enroot list | grep "pyxis_${TARGET_TAG}" | head -n 1 )"
while ! [ -z "${NEXT_CONTAINER}" ]; do
    echo "Deleting ${NEXT_CONTAINER}."
    enroot remove -f ${NEXT_CONTAINER}
    NEXT_CONTAINER="$( enroot list | grep "pyxis_${TARGET_TAG}" | head -n 1 )"
done

echo "Done."
