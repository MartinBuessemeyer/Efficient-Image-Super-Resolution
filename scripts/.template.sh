#!/usr/bin/env bash

SCRIPT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
. "${SCRIPT_ROOT}/.include.sh"
apply_config_files

function usage() {
    echo "./scripts/MYSCRIPT MYARG [...]"
    echo "this is a template script as a basis for new scripts"
    echo
    echo "    MYARG - this argument does something"
    # TODO: adapt usage info
}

function config_variables_required() {
    # TODO: adapt required variables
    variable_required ENROOT_IMAGE_HOME    "this is where your enroot images (.sqsh files) are stored"
}

if ! [ "$#" = "1" ] || [ "${1}" = "-h" ]; then
    # argument is not equal to 1
    usage_with_required_variables
    exit
fi

config_variables_required

MYARG="${1}"
shift  # this removes the (current) first argument.

echo "Do something that MYSCRIPT is supposed to do. MYARG is: \"${MYARG}\""
echo "The optional arguments are:"
echo "$@"
