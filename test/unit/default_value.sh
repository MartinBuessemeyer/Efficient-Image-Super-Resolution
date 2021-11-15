#!/usr/bin/env bash


test_def "regular assign"
(
    default_value "FOO" foo
    assert_equals "${FOO}" "foo"
)
test_end


test_def "assign with ="
(
    default_value BAR=bar
    assert_equals "${BAR}" "bar"
)
test_end


test_def "assign with = and ''"
(
    default_value BAR='bar'
    assert_equals "${BAR}" "bar"
)
test_end


test_def "does not overwrite"
(
    default_value var first_val
    default_value var other_val
    assert_equals "${var}" "first_val"
)
test_end


test_def "assign value with spaces"
(
    default_value "FOOBAR" "foo bar"
    assert_equals "${FOOBAR}" "foo bar"
)
test_end


test_def "assign value with = and spaces"
(
    default_value FOOBAR="foo bar"
    assert_equals "${FOOBAR}" "foo bar"
)
test_end


test_def "assign value for current users works"
(
    default_value_for_user "${USER}" FOOBAR="foo bar"
    assert_equals "${FOOBAR}" "foo bar"
)
test_end


test_def "assign value for other users does not work"
(
    default_value_for_user "Xx${USER}xX" FOOBAR="foo bar"
    assert_equals "${FOOBAR}" ""
)
test_end
