#!/usr/bin/env bash


test_def "it should set the variable"
(
    FOOBAR="foo bar"
    output="$( echo_variable_save FOOBAR )"
    assert_equals "${output}" "FOOBAR=\"foo bar\""
)
test_end


test_def "strings with quotes"
(
    FOOBAR="'foo bar'"
    output="$( echo_variable_save FOOBAR )"
    assert_equals "${output}" "FOOBAR=\"'foo bar'\""
)
test_end
