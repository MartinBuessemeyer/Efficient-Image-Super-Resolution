#!/usr/bin/env bash

function usage() {
    echo "./scripts/.internal-slurm-run.sh"
    echo "    This script should not be started manually and only be run by slurm (or within a slurm allocation)."
    echo "    Please check out submit.sh instead."
}

if [ -z "${SLURM_JOB_ID}" ]; then
    usage
    exit
fi

. "${SCRIPT_ROOT}/.include.sh"
apply_config_files

# do some basic checks about and preparations for the experiment first
slurm_run_common

# default values:
default_value CONTAINER_WORKDIR "/workspace"  # the working directory in enroot
default_value LOG_FILE "${EXP_DIR}/logs/training.log"

echo "> Starting the experiment..."

slurm_run_pyxis -o "${LOG_FILE}" /bin/bash /workspace/run.sh

echo "> Experiment finished."
