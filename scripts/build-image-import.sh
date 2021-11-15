#!/usr/bin/env bash

SCRIPT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
. "${SCRIPT_ROOT}/.include.sh"
apply_config_files

function usage() {
    echo "./scripts/build-image-import.sh"
    echo "directly imports the base image defined by BASE_IMAGE from docker (unchanged)"
}

function config_variables_required() {
    variable_required ENROOT_IMAGE_HOME    "this is where your enroot images (.sqsh files) are stored"
    variable_required TARGET_TAG           "output enroot tag"
    variable_required TARGET_SQSH          "output enroot .sqsh file"
    variable_required BASE_IMAGE           "the base image from docker"
}

if ! [ "$#" = "0" ] || [ "${1}" = "-h" ]; then
    usage_with_required_variables
    exit
fi

config_variables_required

(
    cd_to_enroot_image_home

    if file_needs_to_be_created "${TARGET_SQSH}"; then
        check_error enroot import -o "${TARGET_SQSH}" "${BASE_IMAGE}"
    fi

    echo "> Done."
)
