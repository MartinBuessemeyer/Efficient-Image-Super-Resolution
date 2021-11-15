#!/usr/bin/env bash

# set SCRIPT_ROOT:
SCRIPT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# set PROJECT_ROOT and make sure that is our working directory
PROJECT_ROOT="$(readlink -f "${SCRIPT_ROOT}/..")"
cd "${PROJECT_ROOT}"

# in this script it is used to:
# - detect config files in "${PROJECT_ROOT}/config"
# - always place experimental results in "${PROJECT_ROOT}/run"

function default_values() {
    # default values (can be overwritten in config files or via environment variables):
    default_value VERBOSITY            0                     # setting VERBOSITY to 1 will make some commands add more output
    default_value DEFAULT_NAME         "unnamed"             # default NAME of an experiment, if no NAME is given
    default_value DEFAULT_SERIES       "default"             # default SERIES of an experiment, if no SERIES is given
    default_value RESULTS_ROOT_DIR     "runs"                # default directory in which experimental results are stored
    default_value NO_DEFAULT_ARGS      "false"               # whether to disable default_args

    default_value SIZE_WARNING_MB      10                    # maximum size in MB before experiment preparation warns about the size of src

    default_value BASE_TAG             "${TARGET_TAG}_base"  # enroot tag of base image
    default_value BASE_SQSH            "${BASE_TAG}.sqsh"    # .sqsh file name of base image
    default_value KEEP_BASE_SQSH       "false"               # whether to keep the base SQSH file after building, true or false

    default_value DOCKER_TAG           "${TARGET_TAG}"       # image tag for the docker image

    default_value CONTAINER_WORKDIR    "/workspace"          # the working directory in the enroot container
    default_value NUM_GPU              2                     # number of GPUs which need to be reserved from slurm
    default_value JOB_NAME             "${TARGET_TAG}"       # name of the JOB passed to slurm (shown for example by the "squeue" command)

    default_value PREF_SHELL           zsh                   # shell used for the interactive session
}

function check_yes() {
    # asks the given yes or no question, returns true if they answer YES
    # usage:
    # if check_yes "Do you really want to delete foo?"; then
    #     rm foo
    # fi

    local prompt="${1}"
    read -p "${prompt} [y/N] " REPLY
    echo ""
    if [[ ! "${REPLY}" =~ ^[Yy]$ ]]; then
        return 1
    fi
    return 0
}

function check_no() {
    # asks the given yes or no question, returns false if they answer NO
    # usage:
    # if check_no "Do you want to exit the script?"; then
    #     exit 0
    # fi

    local prompt="${1}"
    read -p "${prompt} [Y/n] " REPLY
    echo ""
    if [[ "${REPLY}" =~ ^[Nn]$ ]]; then
        return 1
    fi
    return 0
}

function default_value() {
    # sets a default value for a variable, if the variable is not already set
    # usage: default_value FOO "foo"

    local variable="${1}"
    local default_value="${2}"
    local help="${3}"

    if [[ ${variable} =~ "=" ]]; then
        variable=${1%=*}
        default_value=${1#*=}
        help="${2}"
    fi

    if [ -z "${!variable}" ]; then
        eval "${variable}="'${default_value}'""
    fi
}

function default_value_for_user() {
    # sets a default value for a variable for a certain user, if the variable is not already set
    # usage: default_value_for_user username FOO "foo"

    local username="${1}"
    shift
    if [ "${username}" == "${USER}" ]; then
        do_if_verbosity 1 echo "  (user specific default value for user '${username}' applied: $@)"
        default_value "$@"
    fi
}

function echo_variable_save() {
    # prints a line, that sets the given variable to its current value
    # usage: echo_variable_save FOO

    local variable="${1}"
    echo "${variable}=\"${!variable}\""
}

function variable_required() {
    # exits the script if a (config) variable is not set, can be used to "store" documentation about environment variables
    # usage: variable_required FOO "help string"

    local variable="${1}"
    local help="${2}"
    variable_info "${variable}" "${help}"
    if [ -z "${!variable}" ]; then
        >&2 echo "! > Variable "${variable}" needs to be set. Please provide it (e.g. in a config file)."
        if ! [ -z "${help}" ]; then
            >&2 echo "! > It is used to '${help}'."
        fi
        exit 1
    fi
}

function variable_info() {
    # this function can be used to output information about environment variables
    # if PRINT_REQUIRED_VARIABLE_INFO is non-zero, it outputs the help information
    # usage: variable_info FOO "help string"

    local variable="${1}"
    local help="${2}"
    if ! [ -z "${PRINT_REQUIRED_VARIABLE_INFO}" ]; then
        printf '%20s - %s\n' "${variable}" "${help}" | indent
        if ! [ -z "${!variable}" ]; then
            printf '%20s   %s\n' "" "current value: '${!variable}'" | indent
        fi
        return 0
    fi
}

function usage_with_required_variables() {
    usage
    echo
    echo "Environment variables for configuration:" | indent
    PRINT_REQUIRED_VARIABLE_INFO="true" config_variables_required
}

function file_required() {
    # exits the script if a given file file does not exist, with an optional help string.
    # usage: file_required /path/to/some/file.txt "This txt file is needed, please put 'foo bar' in it."

    local filepath="${1}"
    local help="${2}"
    if ! [ -f "${filepath}" ]; then
        >&2 echo "! > File '${filepath}' missing. ${help}"
        exit 1
    else
        echo "> File '${filepath}' found."
    fi
}

function file_needs_to_be_created() {
    # checks if a specific file needs creation
    # usage:
    # if file_needs_to_be_created foo; then
    #     touch foo
    # fi

    local file="${1}"
    if [ -f "${file}" ]; then
        echo "> File "${file}" already exists."
        return 1
    else
        return 0
    fi
}

function do_if_verbosity() {
    # shows and then runs a command
    # usage: do_if_verbosity 1 echo "Verbosity is 1 or higher"

    local verbosity_threshold="${1}"
    shift
    if [ "${VERBOSITY:-0}" -ge "${verbosity_threshold}" ]; then
        "$@"
    fi
}

function show_and_run() {
    # shows and then runs a command
    # usage: show_and_run mv foo bar

    echo + $@
    "$@"
}

function check_error() {
    # shows and then runs a command. if the exit code is not zero, asks the user whether to continue
    # usage: check_error mv foo bar

    echo + $@
    "$@"
    local exit_code=$?
    if [ "${exit_code}" -ne 0 ]; then
        if ! check_yes "! > An error occured, continue with the script?"; then
            exit 1
        fi
    fi
}

function remove_lines() {
    # removes the given number of lines from stdout
    # usage: remove_lines 3

    local remove=${1}
    for i in `seq 1 ${remove}`; do
        tput el;
        tput cuu 1;
    done
    tput el;
}

function print_head_line() {
    # prints a line spanning the terminal width
    # usage: remove_lines 3

    local label="${1}"
    local label_length=${#label}
    local columns=$( tput cols )
    local num_signs=$(( (columns - label_length) / 2 - 2))
    printf '=%.0s' $(seq 1 ${num_signs})
    echo -n " ${label} "
    printf '=%.0s' $(seq 1 ${num_signs})
    echo
}

function interactive_tail_cut() {
    # prints the last few lines of a file repeatedly, without scrolling the terminal (cuts at the width of the terminal)
    # usage: interactive_tail_cut 5 path/to/foo.log foo.log

    local num="${1}"
    local file="${2}"
    local label="${3}"
    print_head_line ${label}
    while true; do
        local columns=$( tput cols )
        { for i in $(seq 1 ${num}); do echo; done; cat "${file}"; } \
            | tail -n ${num} - | sed 's/\t/  /g' | cut -c -${columns};
        sleep 2;
        remove_lines ${num}
    done
}

function indent() {
    # indents the output of another command
    # usage: echo "hello world" | indent

    sed 's/^/    /';
}

function show_file() {
    # shows the content of a file
    # usage: show_file /path/to/file name_shown

    local file_path=${1}
    local file_alias="${file_path}"
    if ! [ -z "${2}" ]; then
        file_alias=${2}
    fi
    echo "============ content of ${file_alias} ============="
    cat "${file_path}" | indent
    echo "============ ${file_alias} (end) ============="
}

function directory_size_in_mb() {
    # get the size of a directory in MB
    # usage: size="$( directory_size_in_mb /path/to/directory )"

    local directory_path="${1}"
    du -s -B M "${directory_path}" | sed 's@^\([0-9]\+\).*@\1@'
}

function read_configfile() {
    # shows and sources a configuration (bash) file in PROJECT_ROOT/config
    # usage: read_file configfile

    local config_file="${1}"
    echo "> Sourcing ${PROJECT_ROOT}/${config_file}..."
    . "${PROJECT_ROOT}/${config_file}"
    do_if_verbosity 1 show_file "${PROJECT_ROOT}/${config_file}" "${config_file}"
}

function apply_config_files() {
    # applies a default config file, machine-specific files and sets missing default values
    # usage: apply_config_files

    if ! [ -z "${CONFIGS_READ_ALREADY}" ]; then
        return 0
    fi
    read_configfile "config/default"
    local host="$(cat /etc/hostname)"
    if [ -f "config/${host}" ]; then
        read_configfile "config/${host}"
    else
        echo "> ( Config file for ${host} not found. If needed, create it at \"config/${host}\". )"
    fi
    CONFIGS_READ_ALREADY="x"
    default_values
    check_disable_default_args
}

function cd_to_enroot_image_home() {
    # cd's into ENROOT_IMAGE_HOME (and creates it if neccessary)

    echo "> enroot images are stored in ENROOT_IMAGE_HOME: ${ENROOT_IMAGE_HOME}"
    if ! [ -d "${ENROOT_IMAGE_HOME}" ]; then
        mkdir -p "${ENROOT_IMAGE_HOME}"
    fi
    check_error cd "${ENROOT_IMAGE_HOME}"
}

function show_brief_exp_status() {
    # output tail of experimental results

    tail "${EXP_DIR}/logs/slurm.out" > "${EXP_DIR}/logs/.temp-tail.log" 2>/dev/null
    show_file "${EXP_DIR}/logs/.temp-tail.log" "slurm output (last 10 lines)"
    tail "${EXP_DIR}/logs/training.log" > "${EXP_DIR}/logs/.temp-tail.log" 2>/dev/null
    show_file "${EXP_DIR}/logs/.temp-tail.log" "training log (last 10 lines)"
}

function check_slurm_submitted_already() {
    # checks if the current experiment was already submitted to slurm to avoid overwriting experiment data

    if [ -f "${EXP_DIR}/.submitted" ]; then
        show_brief_exp_status
        echo "! > It seems this experiment has already been submitted to slurm (please check above log output)."
        echo "    Stored slurm job id: $( cat "${EXP_DIR}/.slurm-job-id" )"
        if [ ${1} = "interactive" ] && check_yes "! > Do you still want to submit this?"; then
            echo "> Still submitting the experiment to slurm."
        else
            echo "> Aborted."
            exit 1
        fi
    fi
}

function check_slurm_started_already() {
    # checks if slurm already started this experiment to avoid overwriting experiment data

    if [ -f "${EXP_DIR}/.started" ]; then
        show_brief_exp_status
        echo "! > It seems this experiment has been started before (please check above log output)."
        echo "    Stored slurm job id: $( cat "${EXP_DIR}/.slurm-job-id" )"
        if [ ${1} = "interactive" ]; then
            if check_yes "! > Do you want to delete the log files and restart?"; then
                rm -rf "${EXP_DIR}/logs/"* "${EXP_DIR}/.started"
            else
                echo "> Aborted."
                exit 1
            fi
        else
            echo "> If you want to rerun this experiment, try to submit it again."
            exit 1
        fi
    fi
}

function enroot_target_sqsh_required() {
    # checks whether the default enroot SQSH image was build already (exits the script if not)
    # usage: enroot_image_exists tag

    file_required "${ENROOT_IMAGE_HOME}/${TARGET_SQSH}" "This is the SQSH file for enroot. Please run the build-image script to create it."
}

function slurm_run_common() {
    # does common checks inside a slurm batch script before running

    if ! [ -d "${EXP_DIR}" ]; then
        echo "Something went wrong, EXP_DIR=\"${EXP_DIR}\" is not a directory"
        echo
        usage
        exit 1
    fi

    echo "> Slurm Job ID: ${SLURM_JOB_ID}"
    echo "> Process ID: $$"
    echo "> Current directory: $(pwd)"
    echo "> Experiment directory (rel): ${REL_EXP_DIR}"
    echo "> Experiment directory (abs): ${EXP_DIR}"

    # make run file executable
    chmod +x "${EXP_DIR}/src/run.sh"

    check_slurm_started_already
    touch "${EXP_DIR}/.started"
    rm -f "${EXP_DIR}/.submitted"

    echo "${SLURM_JOB_ID}" > "${EXP_DIR}/.slurm-job-id"
}

function slurm_run_pyxis() {
    # runs the given command (possibly with args before) through pyxis with mounts in the target image

    enroot_target_sqsh_required

    # first, we check if container_mounts errors:
    MOUNT_POINTS_STRING="$( container_mounts )"
    if ! [ "$?" = "0" ]; then
        echo "! > The function container_mounts did not exit cleanly."
        exit 1
    fi
    # replace new lines from container_mounts with comma
    MOUNT_POINTS_STRING="$( container_mounts | tr '\n' ',' | sed 's/,$/\n/' )"

    # add date and time, the .sqsh file was last modified, to prevent caching of outdated images
    SQSH_MODIFIED="$(date -r "${ENROOT_IMAGE_HOME}/${TARGET_SQSH}" +"%Y_%m_%d__%H_%M_%S")"

    show_and_run srun \
        --container-image="${ENROOT_IMAGE_HOME}/${TARGET_SQSH}" \
        --container-name="${TARGET_TAG}_${SQSH_MODIFIED}" \
        --container-mounts="${MOUNT_POINTS_STRING}" \
        --container-workdir="${CONTAINER_WORKDIR}" \
        "$@"
}

function check_experiment_dir_size() {
    if [ "$( directory_size_in_mb "${EXP_DIR}" )" -gt "${SIZE_WARNING_MB}" ]; then
        echo "! > Warning: Your experiment code prepared in ${REL_EXP_DIR} is larger than ${SIZE_WARNING_MB} MB."
        (
            echo "( This seems quite a lot for code, which can fill up the harddisk space quickly. )"
            echo "( Can you mount some folders or exclude some files instead of copying them each time? )"
            echo "( This can be done in scripts/customize.sh -> prepare_experiment_dir or by setting an exclude list. )"
            echo "( To suppress this warning, set SIZE_WARNING_MB=${SIZE_WARNING_MB} to a higher value. )"
        ) | indent
    fi
}

function select_and_prepare_experiment_dir() {
    # selects a directory path for an experiment based on current variables and runs prepare_experiment_dir
    # output variables:
    # EXP_DIR     - absolute path to experiment directory
    # REL_EXP_DIR - relative path to experiment directory

    if [ -z "${SERIES}" ]; then
        SERIES="${DEFAULT_SERIES}"
        echo "> Using '${SERIES}' series. (Set SERIES to change this.)"
    fi
    if [ -z "${NAME}" ]; then
        NAME="${DEFAULT_NAME}"
        echo "! > Using experiment name '${NAME}'."
        echo "! > Please provide a meaningful name in the environment variable NAME to change this."
    fi
    local TIMESTAMP="$(date +"%Y_%m_%d__%H_%M_%S")"
    REL_EXP_DIR="${RESULTS_ROOT_DIR}/${SERIES}/${NAME}/${TIMESTAMP}"
    EXP_DIR="${PROJECT_ROOT}/${REL_EXP_DIR}"

    echo "> Preparing experiment in: ${REL_EXP_DIR}"
    prepare_experiment_dir "$@" | indent

    # create or update the link to the most recent run:
    ln -snfv "${TIMESTAMP}" "${RESULTS_ROOT_DIR}/${SERIES}/${NAME}/last"
}

function check_disable_default_args() {
    # checks whether default_args should be disabled and if so overwrites it

    if [ ${NO_DEFAULT_ARGS} = "true" ]; then
        function default_args() {
            echo ""
        }
    fi
}


# the following functions should be OVERWRITTEN in customize.sh
# a basic version is provided here:

# OVERWRITTEN by customize.sh
function prepare_experiment_dir() {
    # this function prepares the experiment directory
    # EXP_DIR - absolute path to the folder where everything regarding one experiment should be stored
    # $@ contains the command that should be run in the experiment

    # create logs directory
    mkdir -p "${EXP_DIR}/logs"
    # copies the src folder, remove -v to make it silent
    rsync -avu --exclude-from="${PROJECT_ROOT}/.copyexclude"  "${PROJECT_ROOT}/src" "${EXP_DIR}"
    # stores the command given by the user to be run later
    echo "$@" $(default_args) > "${EXP_DIR}/src/run.sh"
    # make a symlink so the command is easier to find
    ln -s "${EXP_DIR}/src/run.sh" "${EXP_DIR}/experiment.sh"
}
# OVERWRITTEN by customize.sh
function default_args() {
    # this function can output some default args (in one line!) that should be used in every experiment
    # they will go AFTER the arguments given by the user
    echo ""
}
# OVERWRITTEN by customize.sh
function docker_build() {
    # this function should create a Docker image with the tag ${DOCKER_TAG}
    docker build -t ${DOCKER_TAG} --build-arg BASE_IMAGE=${BASE_IMAGE} .
}
# OVERWRITTEN by customize.sh
function container_mounts() {
    # this function should output one container mount point per line in the pyxis format:
    # echo SRC:DST[:FLAGS]
    # echo SRC2:DST2[:FLAGS2]
    echo "${EXP_DIR}/src":"/workspace"
    echo "${EXP_DIR}/logs":"/logs"
}
. "${SCRIPT_ROOT}/customize.sh"
