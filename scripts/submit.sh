#!/usr/bin/env bash

SCRIPT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
. "${SCRIPT_ROOT}/.include.sh"
apply_config_files

function usage() {
    echo "./scripts/submit.sh EXP_DIR [ARGS_FOR_SLURM...]"
    echo "    Run a prepared experiment with sbatch (on the training partition)."
    echo
    echo "    EXP_DIR        - the directory to the prepared experiment"
    echo "    ARGS_FOR_SLURM - these args are passed to the sbatch command."
}

function config_variables_required() {
    variable_required NUM_GPU     "number of GPUs which need to be reserved from slurm"
    variable_required JOB_NAME    "name of the JOB passed to slurm (shown for example by the \"squeue\" command)"
}

if [ "$#" = "0" ]; then
    usage_with_required_variables
    exit
fi

config_variables_required
enroot_target_sqsh_required

REL_EXP_DIR="${1}"
EXP_DIR="$( readlink -f "${1}")"
shift  # this removes the (current) first argument (this is important!)

if ! [ -d "${EXP_DIR}" ]; then
    echo "! > EXP_DIR \"${EXP_DIR}\" does not seem to be a directory."
    usage
    exit 1
fi

check_slurm_submitted_already interactive
check_slurm_started_already interactive

SLURM_RUN_FILE="${EXP_DIR}/slurm.sh"

(
    echo "#!/usr/bin/env bash"
    echo
    echo_variable_save EXP_DIR
    echo_variable_save REL_EXP_DIR
    echo_variable_save PROJECT_ROOT
    echo_variable_save SCRIPT_ROOT
    echo_variable_save NUM_GPU
    echo
    echo ". \${SCRIPT_ROOT}/.internal-slurm-run.sh"
) > "${SLURM_RUN_FILE}"

do_if_verbosity 1 show_file "${SLURM_RUN_FILE}"

touch "${EXP_DIR}/logs/training.log"
touch "${EXP_DIR}/logs/slurm.out"

touch "${EXP_DIR}/.submitted"

EMAIL_ARGS=""
if ! [ -z "${EMAIL}" ] || [ "${1}" = "-h" ]; then
    EMAIL_ARGS="--mail-type ALL --mail-user ${EMAIL}"
fi

echo
show_and_run sbatch -p training \
    ${EMAIL_ARGS} \
    --gres gpu:${NUM_GPU} \
    -o "${EXP_DIR}/logs/slurm.out" \
    -J "${JOB_NAME}" \
    "$@" \
    "${SLURM_RUN_FILE}"
echo

echo "> Submitted experiment in \"${REL_EXP_DIR}\"."
echo "> You can watch the training output with:"
echo "$ ./scripts/watch-logs.sh \"${REL_EXP_DIR}\"" | indent
