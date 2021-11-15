#!/usr/bin/env bash

SCRIPT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
. "${SCRIPT_ROOT}/.include.sh"
apply_config_files

function usage() {
    echo "./scripts/build-image-docker.sh"
    echo "builds and imports the docker image into enroot"
}

function config_variables_required() {
    variable_required ENROOT_IMAGE_HOME    "this is where your enroot images (.sqsh files) are stored"
    variable_required TARGET_TAG           "output enroot tag"
    variable_required TARGET_SQSH          "output enroot .sqsh file"
    variable_required BASE_IMAGE           "the base image from docker"

    variable_required DOCKER_TAG           "docker image tag"
}

if ! [ "$#" = "0" ] || [ "${1}" = "-h" ]; then
    usage_with_required_variables
    exit
fi

config_variables_required

# remove default docker:// prefix:
BASE_IMAGE=${BASE_IMAGE#"docker://"}

# the function docker_build is defined in ./scripts/customize.sh
docker_build

(
    cd_to_enroot_image_home

    if file_needs_to_be_created "${TARGET_SQSH}"; then
        check_error enroot import -o "${TARGET_SQSH}" "dockerd://${DOCKER_TAG}"
    fi

    echo "> Done."
)
