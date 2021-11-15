#!/usr/bin/env bash

SCRIPT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
. "${SCRIPT_ROOT}/.include.sh"
apply_config_files

function usage() {
    echo "./scripts/watch-logs.sh EXP_DIR"
    echo "    Shows the tail of all files in the logs directory for the experiment EXP_DIR."
}

if ! [ "$#" = "1" ] || [ "${1}" = "-h" ]; then
    usage
    exit
fi

REL_EXP_DIR="${1}"
EXP_DIR="$( readlink -f "${1}")"

if ! [ -d "${EXP_DIR}" ]; then
    echo "! > EXP_DIR \"${EXP_DIR}\" does not seem to be a directory."
    usage
    exit 1
fi

watch tail "${EXP_DIR}/logs/"*
