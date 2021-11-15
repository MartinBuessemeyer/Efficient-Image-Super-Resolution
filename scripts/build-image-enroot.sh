#!/usr/bin/env bash

SCRIPT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
. "${SCRIPT_ROOT}/.include.sh"
apply_config_files

function usage() {
    echo "./scripts/build-image-enroot.sh"
    echo "imports the base image into enroot, runs the installation, and saves it as a SQSH file"
}

function config_variables_required() {
    variable_required ENROOT_IMAGE_HOME    "this is where your enroot images (.sqsh files) are stored"
    variable_required TARGET_TAG           "output enroot tag"
    variable_required TARGET_SQSH          "output enroot .sqsh file"
    variable_required BASE_IMAGE           "the base image from docker"

    variable_required BASE_TAG             "enroot tag of base image"
    variable_required BASE_SQSH            ".sqsh file name of base image"
    variable_required KEEP_BASE_SQSH       "whether to keep the base SQSH file after building, true or false"
    variable_required INSTALLATION_SCRIPT  "the script to run for installation (in the scripts folder)"
}

if ! [ "$#" = "0" ] || [ "${1}" = "-h" ]; then
    usage_with_required_variables
    exit
fi

config_variables_required

function install() {
    check_error \
        enroot start ${INSTALL_CONTEXT} \
            --rw \
            -m "${PROJECT_ROOT}/install":/install \
            -- "${BASE_TAG}" \
            bash /install/install.sh "$@"
}

function install_root() {
    INSTALL_CONTEXT="--root" install "$@"
}

function install_user() {
    INSTALL_CONTEXT="" install "$@"
}

function run_enroot_installation() {
    install_root "${INSTALLATION_SCRIPT}"
}

function import_base_image() {
    if file_needs_to_be_created "${BASE_SQSH}"; then
        check_error enroot import -o "${BASE_SQSH}" "${BASE_IMAGE}"
    fi
}

function create_base_image() {
    import_base_image
    check_error enroot create -f "${BASE_SQSH}"
}

function build_target_image() {
    if file_needs_to_be_created "${TARGET_SQSH}"; then
        create_base_image

        run_enroot_installation

        check_error enroot export -f --output "${TARGET_SQSH}" "${BASE_TAG}"
        check_error enroot remove -f "${BASE_TAG}"
        if ! [ "${KEEP_BASE_SQSH}" = "true" ]; then
            check_error rm "${BASE_SQSH}"
        fi
    fi
}

(
    cd_to_enroot_image_home

    build_target_image

    echo "> Done."
)
