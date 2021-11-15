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

rm -f .test.log

FAILED=0
PASSED=0
SKIPPED=0

for folder in "unit" "workflow"
do
    echo "Folder ${folder}:"
    for test_sh in $( ls ${TEST_ROOT}/${folder} )
    do
        if ! [ -f "${TEST_ROOT}/${folder}/${test_sh}" ]; then
            continue
        fi
        echo -n "# Testing "
        printf '%-40s : ' "${folder}/${test_sh}"
        . ${TEST_ROOT}/${folder}/${test_sh}
        echo
    done
done

if [ -f .test.log ]; then
    echo
    echo "== Test Log =="
    cat .test.log
    echo
fi

echo
echo "==== Summary ===="
echo "Passed: ${PASSED}, Skipped: ${SKIPPED}"
echo "Failed: ${FAILED}"
echo
