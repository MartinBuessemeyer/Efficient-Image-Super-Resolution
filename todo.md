## TODO LIST

...to make this your own project:

- [ ] Clone this template (you probably do not want to fork it!)
- [ ] Adapt the default config file `config/default`:
    - [ ] Adapt `TARGET_TAG` (give it a cool and unique project name)
    - [ ] Adapt `EMAIL` to yours (and uncomment the line), so you get e-mail updates from slurm
    - You can adapt other values later, but here are some more suggestions:
        - [ ] Adapt `ENROOT_IMAGE_HOME` to share images with other users
        - [ ] Adapt `VERBOSITY` to show sourced config files to help debugging
- [ ] Decide how you want to get your image into enroot:
    - Build with enroot (recommended since you do not need a docker installation):
        - [ ] Configure `BASE_IMAGE` if needed
        - [ ] Adapt `00_install.sh` to install your requirements, the `install` folder
          is mounted during installation (to `/install`), so you can use it
          to store extra files (e.g. requirements.txt)
    - Build with docker:
        - [ ] Configure `BASE_IMAGE` with the docker image you want to start from
        - [ ] Adapt the `Dockerfile` so it installs your requirements
    - Import from docker:
        - [ ] Configure `BASE_IMAGE` with the docker image you want to import
- [ ] Delete the *src* folder (`rm -rf src`) and put your own model and training code there
    - If you have your code in a git repository already, you can simply clone your repository:
      ```bash
      git clone git@github.com:user/project.git src
      ```
    - Alternatively, you can also add it as a [submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules), e.g. with:
      ```bash
      git add src  # we need to remove the previous src folder from index first
      git submodule add git@github.com:user/project.git src
      ```
      This couples your experiment scripts with your source code changes (this often - but not always - makes sense).
      In this case, make sure to clone your project with `--recursive` in the future.
- [ ] Adapt functions in [customize.sh](scripts/customize.sh):
    - [ ] If needed, adapt `prepare_experiment_dir` if you need to copy additional files into each experiment (see the comments there).
      To exclude large files from being copied, adapt [.copyexclude](.copyexclude).
      You can use asterisk wildcards like `*.zip`.
    - [ ] If needed, adapt `default_args` to change which arguments are equal for all of your experiments (see the comments there).
    - [ ] If needed, adapt `container_mounts` to change which folders are mounted into the enroot container (see the comments there).
    - [ ] If needed, adapt `docker_build` or delete it if you do not plan to build your image with docker.
- [ ] Check out the [Customization](README.md#customization) section for more documentation on variables and functions
- [ ] Adapt the README.md for your own project
- [ ] Remove (or rename) the *git remote* and configure it for your newly created project.
  You might as well commit your changes afterwards and push it to your new project.
  ```bash
  git remote remove origin
  #git remote rename origin project-template
  git remote add origin git@gitlab.hpi.de:username/my-awesome-project.git
  ```

Finally you **can** remove some or all of the following files to clean up:

- [ ] Replace the symlink `./scripts/build-image.sh` and remove the unneeded `./build-*.sh` files.
    For example if you want to use enroot (good choice!),
    you really only need `build-image-enroot.sh`.
    ```bash
    mv ./scripts/build-image-enroot.sh ./scripts/build-image.sh && rm scripts/build-image-*.sh
    ```
- [ ] If you do **not** use docker to build your image: you can delete the [Dockerfile](Dockerfile)
- [ ] If you do **not** use enroot to build your image: you can delete the [install](install) folder
- [ ] Machine specific config files if you do not need them right now (they are easy to recreate later)
- [ ] The [test](test) folder
- [ ] The file [demo.md](demo.md)
- [ ] This file [todo.md](todo.md)
