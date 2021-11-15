#!/usr/bin/env bash

SCRIPT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
. "${SCRIPT_ROOT}/.include.sh"
apply_config_files

function usage() {
    echo "./scripts/run.sh HOST EXP_DIR [ARGS_FOR_SLURM...]"
    echo "    Run a prepared experiment with srun (on the gpu partition). (Intended for shorter/debug sessions.)"
    echo
    echo "    HOST           - the host you want to work on, use 'localhost' to run locally"
    echo "                     (e.g. you already are logged into the debug host with the correct GPU allocation)"
    echo "    EXP_DIR        - the directory to the prepared experiment"
    echo "    ARGS_FOR_SLURM - these args are passed to the srun command."
}

function config_variables_required() {
    variable_required NUM_GPU    "number of GPUs for the run (no effect if working locally)"
    variable_required JOB_NAME   "name of the JOB passed to slurm (shown for example by the \"squeue\" command)"
}

if [ "$#" -lt "2" ] || [ "${1}" = "-h" ]; then
    usage_with_required_variables
    exit
fi

config_variables_required
enroot_target_sqsh_required

HOST="${1}"
shift

REL_EXP_DIR="${1}"
EXP_DIR="$( readlink -f "${1}")"
shift

if ! [ -d "${EXP_DIR}" ]; then
    echo "! > EXP_DIR \"${EXP_DIR}\" does not seem to be a directory."
    usage
    exit 1
fi

check_slurm_started_already interactive

slurm_run_common

if ! [ "${HOST}" = "localhost" ]; then
    HOST_ARGS="-w ${HOST} --gres gpu:${NUM_GPU}"
    read_configfile "config/${HOST}"
fi

echo "> Starting the experiment (this can take a while to start!)..."

slurm_run_pyxis ${HOST_ARGS} --pty "$@" /bin/bash /workspace/run.sh

echo "> Run ended."
echo "> Output files are stored in: ${REL_EXP_DIR}"
