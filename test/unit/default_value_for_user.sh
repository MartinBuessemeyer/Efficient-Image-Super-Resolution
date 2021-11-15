#!/usr/bin/env bash


test_def "value is assigned for current user"
(
    default_value_for_user "${USER}" FOOBAR="foo bar"
    assert_equals "${FOOBAR}" "foo bar"
)
test_end


test_def "value is not assigned for other users"
(
    default_value_for_user "Xx${USER}xX" FOOBAR="foo bar"
    assert_equals "${FOOBAR}" ""
)
test_end
