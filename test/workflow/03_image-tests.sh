#!/usr/bin/env bash

skip_on_hosts_other_than slurmsubmit

function overwrite_container_mounts() {
    # overwrite container mounts
    function container_mounts() {
        echo "${PROJECT_ROOT}/test/workflow/image-tests":"/test"
    }
}

echo

(
    apply_config_files

    for tag in "my_project" "my_project_docker"
    do
        TARGET_TAG="${tag}"
        TARGET_SQSH="${TARGET_TAG}.sqsh"
        ENROOT_IMAGE_HOME="${ENROOT_IMAGE_HOME}"
        CONTAINER_WORKDIR="/test"

        overwrite_container_mounts

        echo
        test_def "nvidia-smi works (${tag})"
        (
            assert_cmd_successful slurm_run_pyxis -w ${DEBUG_HOST} -G 1 --pty /bin/bash /test/nvidia-smi.sh
        )
        test_end

        echo
        test_def "pytorch works with cuda (${tag})"
        (
            assert_cmd_successful slurm_run_pyxis -w ${DEBUG_HOST} -G 1 --pty /bin/bash /test/pytorch-cuda.sh
        )
        test_end

    done
)
