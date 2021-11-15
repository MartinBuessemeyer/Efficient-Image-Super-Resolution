#!/usr/bin/env bash

SCRIPT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
. "${SCRIPT_ROOT}/.include.sh"
NO_DEFAULT_ARGS="true" apply_config_files

function usage() {
    echo "./scripts/interactive.sh [EXP_DIR] [-- ARGS_FOR_SLURM...]"
    echo "start an interactive session inside the enroot container on the HOST"
    echo
    echo "    EXP_DIR        - the experiment directory (if you want to work in an existing experiment dir)"
    echo "    ARGS_FOR_SLURM - these args are passed to the srun command, you need to use two hyphens before them: --"
}

function config_variables_required() {
    variable_required PREF_SHELL           "shell used for the interactive session"
    variable_required CONTAINER_WORKDIR    "the working directory in the enroot container"
}

if [ "${1}" = "-h" ]; then
    usage_with_required_variables
    exit
fi

config_variables_required
enroot_target_sqsh_required

separate_session_directory="false"

if [ "$#" = "0" ] || [ "${1}" = "--" ]; then
    separate_session_directory="true"
    shift
else
    REL_EXP_DIR="${1}"
    EXP_DIR="$( readlink -f "${1}")"
    shift
    if ! ( [ "${1}" = "--" ] || [ "$#" = "0" ] ); then
        usage_with_required_variables
        exit
    fi
    shift
fi

if [ "${separate_session_directory}" = "true" ]; then
    select_and_prepare_experiment_dir /bin/${PREF_SHELL}

    # basic checks before running
    slurm_run_common
fi

if ! [ -d "${EXP_DIR}" ]; then
    echo "> ! Experiment directory '${EXP_DIR}' not found."
    exit 1
fi

# set a default SERIES and IMAGENET_HOME for our interactive session:
DEFAULT_SERIES="interactive_session"          # use "interactive_session" as the default series
default_value IMAGENET_HOME="/"               # to debug on a machine without actual data, we set a dummy default path
default_value CONTAINER_WORKDIR "/workspace"  # the working directory in enroot

echo
echo "> Starting the interactive session (this can take a while to start)..."
echo

slurm_run_pyxis --pty "$@" "${PREF_SHELL}"

echo "> Interactive session ended."
echo "> Output files are stored in: ${REL_EXP_DIR}"
echo "> Reminder: release the allocation, once you are finished."
