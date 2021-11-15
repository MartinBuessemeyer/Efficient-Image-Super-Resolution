#!/usr/bin/env bash

SCRIPT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
. "${SCRIPT_ROOT}/.include.sh"
apply_config_files

function usage() {
    echo "./scripts/prepare.sh TRAIN_COMMAND [TRAIN_ARGS...]"
    echo "    Prepare a training experiment to run later on."
    echo "    (Stores a copy of the current code and the given run command in a dedicated folder.)"
}

function config_variables_required() {
    variable_required SIZE_WARNING_MB "maximum size in MB before experiment preparation warns about the size of src"
    variable_info     NAME            "name of the experiment"
    variable_info     SERIES          "series name of the current experiment (e.g. test, ablation, paper_results, ...)"
}

if [ "$#" = "0" ] || [ "${1}" = "-h" ]; then
    usage_with_required_variables
    exit
fi

config_variables_required

select_and_prepare_experiment_dir "$@"

check_experiment_dir_size

echo "> The experiment is prepared, showing folder size, absolute path and stored command:"
echo
du -sh "${EXP_DIR}" | indent
echo "\$ $(cat "${EXP_DIR}/experiment.sh")" | indent

function show_submission_info() {
    echo
    echo "> Submit your experiment to slurm now (or later) with:"
    echo "$ ./scripts/submit.sh \"${REL_EXP_DIR}\"" | indent
}

do_if_verbosity 0 show_submission_info
