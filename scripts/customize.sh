#!/usr/bin/env bash

# possibly useful global variables in all functions:
# PROJECT_ROOT - absolute path to the project root directory
# SCRIPT_ROOT - absolute path to the scripts folder

# here you can customize (overwrite) functions

function prepare_experiment_dir() {
    # this function prepares the experiment directory
    # EXP_DIR - absolute path to the folder where everything regarding one experiment should be stored
    # $@ contains the command that should be run in the experiment

    # create ${EXP_DIR} and logs directory
    mkdir -p "${EXP_DIR}/logs"

    # copy the src folder, remove -v to make it silent
    rsync -avu --exclude-from="${PROJECT_ROOT}/.copyexclude"  "${PROJECT_ROOT}/src" "${EXP_DIR}"

    # stores the command given by the user to be run later and make a symlink to make it easier to find later
    echo "$@" $(default_args) > "${EXP_DIR}/src/run.sh"
    ln -s "${EXP_DIR}/src/run.sh" "${EXP_DIR}/experiment.sh"
}

function default_args() {
    # this function can output some default args (in one line!) that should be used in every experiment
    # they will go AFTER the arguments given by the user

    # always pass the mounted imagenet directory inside enroot
    echo "--imagenet-directory /data"

    # in the default code NUM_RESERVED_GPU is set to the number of GPUs reserved by slurm,
    # so you could use something like this as well to pass the number of GPUs to your training script:
    #echo "--gpus \${NUM_RESERVED_GPU} --imagenet-directory /data"

    # if you want empty default_args, just do echo "" but make sure delete the other 'echos' above:
    #echo ""
}

function container_mounts() {
    # this function should output one container mount point per line in the pyxis format:
    # echo SRC:DST[:FLAGS]
    # echo SRC2:DST2[:FLAGS2]

    echo "${EXP_DIR}/src":"/workspace"
    echo "${EXP_DIR}/logs":"/logs"

    # if a path depends on a variable, we should make sure it is set with variable_required and provide a help string
    variable_required IMAGENET_HOME "path should contain two folders 'val' and 'train' with the corresponding imagenet data"
    echo "${IMAGENET_HOME}":"/data"
}

function docker_build() {
    # this function should create a Docker image with the tag ${DOCKER_TAG}

    # if your Dockerfile does not support changing the base image remove the --build-arg BASE_IMAGE=${BASE_IMAGE}
    docker build -t ${DOCKER_TAG} --build-arg BASE_IMAGE=${BASE_IMAGE} .
}
