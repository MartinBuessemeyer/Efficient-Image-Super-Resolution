#!/usr/bin/env bash

echo

test_def "build image"
(
    skip_on_hosts_other_than slurmsubmit

    export TARGET_TAG="my_project_docker"
    export TARGET_SQSH="${TARGET_TAG}.sqsh"

    assert_cmd_successful srun -w ${DEBUG_HOST} --pty "${TEST_ROOT}/clean-test-setup.sh"

    assert_cmd_successful srun -w ${DEBUG_HOST} --pty ./scripts/build-image-docker.sh
)
test_end
