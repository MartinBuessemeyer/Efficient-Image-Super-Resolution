#!/usr/bin/env bash

INSTALL_DIR="/install"

function usage() {
    echo "${INSTALL_DIR}/install.sh SCRIPT [SCRIPT...]"
    echo "this script provides basic functions for installation and runs the given installation recipe(s)"
    echo "running it outside of your enroot container might lead to unwanted results"
    echo
    echo "    SCRIPT - the name of a script in the install folder"
}

function RUN() {
  echo "### Running: $@"
  "$@"
  CODE=$?
  if [ "${CODE}" -ne 0 ]; then
      echo "!!! [Errored]: \"$@\""
      exit 1
  else
      echo "### [Success]: \"$@\""
  fi
}

# usage: WORKDIR /some/directory
# this creates the given directory, navigates to it and tries to configure the container to start in that directory
function WORKDIR() {
    mkdir -p "${1}" && cd "${1}"
    # this permanently changes the working directory (if /etc/rc.local is sourced during startup)
    (
        echo "# automatically overwritten by WORKDIR command"
        echo "mkdir -p \"\${WORK_DIR}\" 2> /dev/null"
        echo "cd \"\${WORK_DIR}\" && unset OLDPWD || exit 1"
    ) > /etc/rc.local
}

for step in "$@"
do
    echo "### Installing: ${step}"
    . ${INSTALL_DIR}/${step}
done
