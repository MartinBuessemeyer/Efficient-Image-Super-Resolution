# Customizable Project Template with Support for SLURM

This is a project template that demonstrates how to use a deep learning package,
such as PyTorch with enroot and slurm and organize experimental results.
It can provide a starting ground for your own project.
Different steps to help you set up your project are already prepared in bash scripts.

If you do not know what this project does, check out the [Demo](demo.md).

If you want to use this template, please check out the [TODO list](todo.md)
to find out how to configure this template to start or integrate your own project.

## Contents

The contents are organized in the following way:

- config file folder: [config](config)
- installation scripts: [install](install)
- management scripts: [scripts](scripts)
- your actual training and model code: [src](src)
- experimental results, model checkpoints, and log files: [runs](runs)

## Configuration

The [config](config) folder contains configuration files.
The [default](config/default) config file is always sourced before every script.

In addition, you can create one file per host, that matches the exact hostname
to provide host specific configuration.
You can see an example of configuring the data directory `IMAGENET_HOME` on two
different machines with hostnames `fb10dl07` (in [config/fb10dl07](config/fb10dl07)) and
`fb10dl09` (in [config/fb10dl09](config/fb10dl09)).

See the full list of variables and how to configure them [here](#configuration-variables).

## Installation

Two common ways to run with enroot are:

- start from an image from dockerhub and install dependencies with enroot
- build a local docker image with a Dockerfile and import this into enroot

Both ways are supported by this template (to some extent), you should decide which one to use.

All scripts start from a docker base image (defined by `BASE_IMAGE`) in some way and
create a .sqsh file named `TARGET_SQSH` (default is `${TARGET_TAG}.sqsh`).
You can later check how to configure this in the [TODO list](todo.md).

### Build with docker

The script `./scripts/build-image-docker.sh` builds a docker image based on `BASE_IMAGE`
with the [Dockerfile](Dockerfile) and imports that into an enroot image.

### Build with enroot

The script `./scripts/build-image-enroot.sh` (or in short `./scripts/build-image.sh`) builds 
the enroot image in enroot.
It starts from the configured `BASE_IMAGE` and runs [`00_install.sh`](install/00_install.sh)
(this file is very similar to the [Dockerfile](Dockerfile)).

### Import from docker

If the docker image `BASE_IMAGE` is already built, it can be directly imported with the 
script `./scripts/build-image-import.sh`.

## Running

An experiment can be prepared with:
```bash
NAME="first_test" ./scripts/prepare.sh python main.py -a resnet18 --epochs 1
``` 
The script that should be executed is expected to lie in the `src` directory.  
The command and code are stored for reproducibility.
This makes the experiment code also independent of code changes in the working directory.

You could then run the experiment on node `slurmnode` with 2 GPUs like this:
```bash
NUM_GPU=2 ./submit.sh runs/default/first_test/* -w slurmnode
```

We can also do both steps in one command with [quick-submit](scripts/quick-submit.sh):
```bash
NAME="first_test" NUM_GPU=2 ./scripts/quick-submit.sh -w slurmnode -- python main.py -a resnet18 --epochs 1
```

### Escaping spaces

You should probably avoid having spaces in your arguments (or file paths),
but you can work around them by escaping them in the following way:
```bash
NAME="first_test" NUM_GPU=0 ./scripts/quick-submit.sh -w slurmnode -- echo "'Hello ${USER}'"
```
This stores `echo 'Hello joseph' --imagenet-directory /data` in the `experiment.sh` (because the
default argument `--imagenet-directory /data` is defined in the [customize script](scripts/customize.sh)).

### Caching of images

The image is cached by `enroot` based on the date and time the *.sqsh* file was last *modified*.
To prevent overly high disk usage, remove old images from time to time on your hosts.
You can check and remove them with:
```bash
$ srun -w slurmhost enroot list
pyxis_my_project_2021_07_06__09_08_32
pyxis_my_project_2021_07_09__09_44_32
$ srun -w slurmhost enroot remove -f pyxis_my_project_2021_07_06__09_08_32
```
(If you remove all images, it will create the image the next time before you run.)

## Debugging

Most scripts should output what they do before they do it, and already have some error
detection included.

You can also export the variables defined in config files into your own bash:
```bash
. ./scripts/activate.sh
```

You can also add `VERBOSITY=1` to your config, which will always output the sourced config files and add other output.

### Interactive session

The script [interactive.sh](scripts/interactive.sh) can be used to interactively test your container image with a `zsh` or `bash` shell.
(But it does not store the console output!)

To start an interactive session we first need to manually get a slurm allocation in which we can work in, e.g. with 1 GPU:
```bash
salloc --gres gpu:1
```

This should connect us to the host already.
(If we need port forwarding, e.g. for debugging, we probably need to connect separately to the allocated machine with port forwarding activated.)
Now we can enter the enroot container and start the interactive session with:
```bash
./scripts/interactive.sh
```
This will create a new clean directory and set environment variables similar to experiments being run with [submit.sh](scripts/submit.sh) or [run.sh](scripts/run.sh).
If we do want to run in the project folder itself, e.g. for debugging, or (re-)use an existing experiment directory, e.g., to finish an analysis, simply pass the folder instead:
```bash
./scripts/interactive.sh .  # use the project directory and ./src/ itself (be careful about deleting/clobbering your files)
./scripts/interactive.sh runs/default/unnamed/2021_08_17__17_02_28  # use an existing experiment folder
```
Within we can run any command inside the enroot container:
```bash
python3
```

### Remote debugging with PyCharm

(This section is based on our [our cluster documentation](https://gitlab.hpi.de/deeplearning/students/cluster-usage-documentation/-/blob/master/enroot/debugging.md).)

This project template already contains:

- Installation of the pip package `pydevd-pycharm` (version of the pip package might need to be adapted - see below)
- The debug code snippet (first three lines) in our [main.py](src/main.py)

However, we still need to:

- Create a _Python Debug Server_ Run/Debug configuration in our local PyCharm instance
- Check if the version `pydevd-pycharm` version in the installation script (`install/00_install.sh`) or the Dockerfile matches the one displayed in the window of the Python Debug Server configuration. If not:
    - copy the pip command from the window of the Python Debug Server configuration and replace it in the installation script or the Dockerfile
    - rebuild the enroot image
- Find our debug port number and set it as `REMOTE_PYCHARM_DEBUG_PORT` in our [default config](config/default)

Check out our [cluster documentation](https://gitlab.hpi.de/deeplearning/students/cluster-usage-documentation/-/blob/master/enroot/debugging.md) on how to do this.

Afterwards we can allocate slurm resources, connect with remote port forwarding and debug our code:

1. Allocate resources on our slurm submission host (replace `DEBUG_HOST` with one of the machines in our cluster):
    ```bash
    $ salloc -G 1 -w DEBUG_HOST
    ```
2. On your local machine, connect to our `DEBUG_HOST` with port forwarding (replace `PORT` with the port configured in our config, `USERNAME` with your cluster username and `DEBUG_HOST` with the machine from the command above. Use the IP address for the `DEBUG_HOST` or you might run into troubles when you are connected via VPN (snx).):
    ```bash
    $ ssh USERNAME@DEBUG_HOST -R PORT:localhost:PORT
    ```
3. Start our _Python Debug Server_ in your local PyCharm
4. Start our program (in the ssh session from step 2). Two examples:
   1) With an interactive session:
    ```bash
    $ REMOTE_PYCHARM_DEBUG_SESSION=true ./scripts/interactive.sh
    $ python main.py -a resnet18 --epochs 1
    ```
   2) With a prepared experiment and the [run](scripts/run.sh) script:
    ```bash
    $ SERIES="debug" NAME="one_epoch" ./scripts/prepare.sh python main.py -a resnet18 --epochs 1
    [...]
    $ REMOTE_PYCHARM_DEBUG_SESSION=true ./scripts/run.sh localhost runs/debug/one_epoch/last
    [...]
    ```

## Customization 

There are three ways to customize this template:

1. Adapt [configuration variables](#configuration-variables)
2. Adapt [functions in customize.sh](#functions)
3. Adapt the scripts directly

A summary of what most likely needs adaptation is in the [TODO list](todo.md).

### Configuration Variables

| config variable     | Meaning                                                                                | Default                                                  | Used in scripts        |
| --------------------| -------------------------------------------------------------------------------------- | -------------------------------------------------------- | ---------------------- |
| VERBOSITY           | verbosity level (setting this to 1 will increase output, e.g. print config files)      | 0                                                        | all                    |
| ENROOT_IMAGE_HOME   | this is where your enroot images (.sqsh files) are stored                              | "/enroot_share/${USER}"                                  | almost all             |
| BASE_IMAGE          | the docker base image for the build script(s)                                          | "docker://pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime" | build-image-*          |
| TARGET_TAG          | output enroot tag                                                                      | "my_project"                                             | build-image-*          |
| TARGET_SQSH         | output enroot .sqsh file                                                               | "${TARGET_TAG}.sqsh"                                     | almost all             |
| BASE_TAG            | enroot tag of base image                                                               | "${BASE_TAG}.sqsh"                                       | build-image-enroot     |
| BASE_SQSH           | the .sqsh file name of base image                                                      | "${BASE_TAG}.sqsh"                                       | build-image-enroot     |
| KEEP_BASE_SQSH      | whether to keep the base SQSH file after building, true or false                       | "false"                                                  | build-image-enroot     |
| INSTALLATION_SCRIPT | the script to run for installation with enroot                                         | "00_install.sh"                                          | build-image-enroot     |
| DOCKER_TAG          | image tag for the docker image                                                         | "${TARGET_TAG}"                                          | build-image-docker     |
| NAME                | name of the current experiment                                                         | *not set*                                                | prepare                |
| SERIES              | series of the current experiment                                                       | "default"                                                | prepare                |
| DEFAULT_NAME        | default name of the current experiment (if none is given)                              | "unnamed"                                                | prepare                |
| DEFAULT_SERIES      | default series of the current experiment (if none is given)                            | "default"                                                | prepare                |
| RESULTS_ROOT_DIR    | default directory in which experimental results are stored                             | "runs"                                                   | prepare                |
| SIZE_WARNING_MB     | maximum size of src folder before a warning appears                                    | 10                                                       | prepare                |
| NO_DEFAULT_ARGS     | if set to `true` it disables default args defined by default_args                      | false                                                    | prepare                |
| EMAIL               | if set, send e-mail updates from jobs to that EMAIL (e.g. `firstname.lastname@hpi.de`) | *not set*                                                | submit                 |
| NUM_GPU             | number of GPUs which need to be reserved from slurm                                    | 2                                                        | submit                 |
| JOB_NAME            | name of the job passed to slurm (shown for example by the "squeue" command)            | "${TARGET_TAG}"                                          | submit                 |
| PREF_SHELL          | preferred shell for interactive sessions, e.g., bash or zsh                            | zsh                                                      | interactive            |
| CONTAINER_WORKDIR   | default working directory inside the enroot container                                  | "/workspace"                                             | run                    |
| IMAGENET_HOME       | imagenet data directory - should always be in a local (SSD or HDD) directory           | *not set*                                                | run                    |

These variables can be configured on the command line directly (before each command), through environment variables,
the [default config file](config/default) or machine-specific config files.

There are four ways to set configuration variables in config files:
```bash
# 1) sets IMAGENET_HOME to /some/path if it is not already set:
default_value IMAGENET_HOME="/some/path"
# 2) overwrite IMAGENET_HOME with /some/path
IMAGENET_HOME="/some/path"
# 3) set ENROOT_ROOTFS_WRITABLE as an environment variable which makes it available to enroot:
export ENROOT_ROOTFS_WRITABLE="yes"
# 4) sets EMAIL to abc@xyz.com if it is not already set but only for user with the username abc:
default_value_for_user abc EMAIL="abc@xyz.com"
```

### Functions

Some code should probably also be customized for your project.
In addition to directly editing the bash files, you can find a few candidates that
might need customization in [customize.sh](scripts/customize.sh).

To exclude large files from being copied with the default preparation code, adapt [.copyexclude](.copyexclude).
You can use asterisk wildcards like `*.zip`.

## Improvements

Please open an issue or merge request to suggest improvements or ask questions.
