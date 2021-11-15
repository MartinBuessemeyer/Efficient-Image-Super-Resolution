#!/usr/bin/env bash

SCRIPT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
. "${SCRIPT_ROOT}/.include.sh"
apply_config_files

function usage() {
    echo "./scripts/quick-submit.sh [ARGS_FOR_SLURM ...] -- TRAIN_COMMAND [TRAIN_ARGS...]"
    echo "    this script calls:"
    echo "     1)      prepare.sh TRAIN_COMMAND [TRAIN_ARGS...]"
    echo "     2)      submit.sh [ARGS_FOR_SLURM ...]"
}

if [ "$#" = "0" ] || [ "${1}" = "-h" ]; then
    usage
    exit
fi

enroot_target_sqsh_required

SLURM_ARGS=()

NEXT_ARG="${1}"
while ! [ "${NEXT_ARG}" = "--" ] && ! [ -z "${NEXT_ARG}" ]; do
    SLURM_ARGS+=( "${NEXT_ARG}" )
    shift
    NEXT_ARG="${1}"
done
shift
if [ -z "$(echo $@)" ]; then
    usage
    exit
fi

VERBOSITY=-1 . ${SCRIPT_ROOT}/prepare.sh "$@"
. ${SCRIPT_ROOT}/submit.sh "${REL_EXP_DIR}" "${SLURM_ARGS[@]}"
