#!/usr/bin/env bash


default_value DEBUG_HOST="fb10dl09"
default_value_for_user joseph DEBUG_EMAIL="joseph.bethge@hpi.de"


function test_def() {
    # define the description of the test
    # used for info and assertions

    CURRENT_TEST_DESC="${1}"
}

function test_end() {
    TEST_RESULT=$?
    if [ ${TEST_RESULT} = "10" ]; then
        echo -n "S"
        SKIPPED=$((SKIPPED + 1))
        return
    fi
    if [ ${TEST_RESULT} = "0" ]; then
        echo -n "."
        PASSED=$((PASSED + 1))
    else
        echo -n "F"
        FAILED=$((FAILED + 1))
    fi
}

function assert_equals() {
    # asserts that two strings are equal
    # assert_equals "${variable}" "expected_value"

    if ! [ "${1}" = "${2}" ]; then
        echo "! Assertion in '${folder}/${test_sh}' - '${CURRENT_TEST_DESC}' failed:" >> .test.log
        echo "    \"${1}\" != \"${2}\" " >> .test.log
        exit 1
    fi
}

function assert_cmd_successful() {
    # asserts that the given command returns an error code of 0

    "$@"
    local ret_code="$?"
    if [ "${ret_code}" -gt "0" ]; then
        echo "! Assertion in '${folder}/${test_sh}' - '${CURRENT_TEST_DESC}' failed:" >> .test.log
        echo "    CMD \"$@\" has returned code ${ret_code}, not 0." >> .test.log
        exit 1
    fi
}

function test_fail() {
    # makes a test fail

    echo "! Fail: '${folder}/${test_sh}' - '${CURRENT_TEST_DESC}'" >> .test.log
    exit 1
}

function test_pass() {
    # makes a test pass

    exit 0
}

function test_skip() {
    # skips a test

    exit 10
}

function skip_on_hosts_other_than() {
    # skips (passes) a test if hostname does not match a string

    local desired_host="${1}"
    local current_host="$(cat /etc/hostname)"
    if [ "${desired_host}" = "${current_host}" ]; then
        return
    fi

    echo "Skipped: '${folder}/${test_sh}' - '${CURRENT_TEST_DESC}'" >> .test.log
    echo "    (not on host '${desired_host}')" >> .test.log
    test_skip
}
