
# Demo
*(You can follow along by running the commands behind the $ symbols if you clone this
repo on your favorite slurm cluster.)*

For this demo we suppose, you are working on your slurm submit host and
can use a machine named `fb10dl07` in your slurm cluster.
(If `fb10dl07` is occupied, the template also supports `fb10dl09`.)

We can build our enroot image on one of the servers (this will take some time) and
afterwards check that it is ready:
```bash
$ srun -w fb10dl07 --pty ./scripts/build-image.sh
> Sourcing /home/joseph/Projects/project-template/config/default...
> Sourcing /home/joseph/Projects/project-template/config/fb10dl07...
> enroot images are stored in ENROOT_IMAGE_HOME: /enroot_share/joseph
+ cd /enroot_share/joseph
+ enroot import -o my_project_base.sqsh docker://pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime
[INFO] Querying registry for permission grant
    [...]
+ enroot start --root --rw -m /home/joseph/Projects/project-template/install:/install -- my_project_base bash /install/install.sh 00_install.sh
    [...]
+ enroot create my_project_base.sqsh
[INFO] Extracting squashfs filesystem...
    [...]
+ rm my_project_base.sqsh
> Done.
$ ls /enroot_share/${USER}
my_project.sqsh
```

Now let us prepare an experiment that runs the `main.py` script located in the `src` directory.
The following command does this for a training of a resnet18 on ImageNet (but for one epoch only so it is quick).

```bash
$ ./scripts/prepare.sh python main.py -a resnet18 --epochs 1
> Sourcing /home/joseph/Projects/project-template/config/default...
> ( Config file for slurmsubmit not found. If needed, create it at "config/slurmsubmit". )
> Using 'default' series. (Set SERIES to change this.)
! > Using experiment name 'unnamed'.
! > Please provide a meaningful name in the environment variable NAME to change this.
> Preparing experiment in: runs/default/unnamed/2021_07_07_15_36_40
    [...]
> The experiment is prepared, showing folder size, absolute path and stored command:

    52K /home/joseph/Projects/project-template/runs/default/unnamed/2021_07_07_15_36_40
    $ python main.py -a resnet18 --epochs 1 --multiprocessing-distributed --imagenet-directory /data

> Submit your experiment to slurm now (or later) with:
    $ ./scripts/submit.sh runs/default/unnamed/2021_07_07_15_36_40
```

A small side note on structuring your experiments with NAME and SERIES
can be seen at the bottom of this page.
But first, let's continue and submit our experiment to slurm on our node `fb10dl07`:

1. Copy the command above (`./scripts/submit.sh runs/default/unnamed/2021_07_07_15_36_40`)
2. Add `NUM_GPU=1` **before** the command (this is an option of our script and 1 GPU is enough for 1 epoch)
3. (Optional) Add `EMAIL=firstname.surname@hpi.de` **before** the command to get email updates from slurm (we only need to do this for the demo)
4. Add `-w fb10dl07` **afterwards** (additional args after the experiment directory are passed to `sbatch`)

The result could look like this:
```bash
$ NUM_GPU=1 EMAIL=joseph.bethge@hpi.de ./scripts/submit.sh runs/default/unnamed/2021_07_07_15_36_40 -w fb10dl07
> Sourcing /home/joseph/Projects/project-template/config/default...
> ( Config file for slurmsubmit not found. If needed, create it at "config/slurmsubmit". )

+ sbatch -p training --mail-type ALL --mail-user joseph.bethge@hpi.de --gres gpu:1 -o /home/joseph/Projects/project-template/runs/default/unnamed/2021_07_08_10_00_55/logs/slurm.out -w fb10dl07 /home/joseph/Projects/project-template/runs/default/unnamed/2021_07_08_10_00_55/slurm.sh
Submitted batch job 925

> Submitted experiment in "/home/joseph/Projects/project-template/runs/default/unnamed/2021_07_08_10_00_55".
> You can watch the training output with:
    $ ./scripts/watch-logs.sh "runs/default/unnamed/2021_07_07_15_36_40"
```

To always get emails in real projects, we can store the variable `EMAIL` permanently
in our [default config](config/default) (do not forget to uncomment the corresponding line).
This is one of the [steps needed to adapt](todo.md) this template for your own projects.
For `NUM_GPU` we could also use the machine specific config files if we need a different number of GPUs because of memory requirements.

While it is training, we can watch the log files with the command shown in the output
(using `./scripts/watch-logs.sh`), or we simply wait until the training finishes
and then check the output:

```bash
$ tail runs/default/unnamed/2021_07_07_15_36_40/logs/slurm.out
> Sourcing /home/joseph/Projects/project-template/config/default...
[...]
> Experiment finished.
$ tail runs/default/unnamed/2021_07_07_15_36_40/logs/training.log
Test: [110/196] Time  0.063 ( 0.583)    Loss 3.5302e+00 (3.6266e+00)    Acc@1  31.64 ( 23.86)   Acc@5  52.73 ( 49.28)
[...]
Test: [190/196] Time  1.139 ( 0.572)    Loss 2.9434e+00 (3.7887e+00)    Acc@1  28.91 ( 22.58)   Acc@5  58.20 ( 46.36)
 * Acc@1 22.912 Acc@5 46.724
```

That's it. Read on with how to structure your experiments below or go straight to the
[TODO list](todo.md) to adapt this project template for your own code.

To run experiments quicker, once you are familiar with what happens in the two steps,
you can use [quick-submit.sh](scripts/quick-submit.sh). For example,
the following calls both `prepare` and `submit` in one call:
```bash
$ NUM_GPU=1 EMAIL=joseph.bethge@hpi.de ./scripts/quick-submit.sh -w fb10dl07 -- python main.py -a resnet18 --epochs 1
```

### A note on structuring your experiments:

This project tries to accustom you to structuring your experiments, as we can read in the output above.

To achieve this, we should at least provide a meaningful NAME to the experiment
(unless it really is a quick experiment like ours that does not need to be stored).

Furthermore, it can make sense to organize experiments in different SERIES,
here are some ideas:
"testing", "hyperparameter_optimization", "ablation_study", "main_paper_results"

You do not need to fear to overwrite existing experiments,
as the current date and time is added to the path at the time of preparing the experiment.

Here are some examples to set a name, or set both a series and a name:
```bash
$ NAME="demo_experiment" ./scripts/prepare.sh python main.py -a resnet18 --epochs 1
$ SERIES="hyperparameter_optimization" NAME="lr_0.01" ./scripts/prepare.sh python main.py -a resnet18 --lr 0.01
$ SERIES="hyperparameter_optimization" NAME="lr_0.02" ./scripts/prepare.sh python main.py -a resnet18 --lr 0.02
```
To change the SERIES, without typing it everytime you could do an export...
```bash
$ export SERIES="hyperparameter_optimization"
$ NAME="lr_0.01" ./scripts/prepare.sh python main.py -a resnet18 --lr 0.01
$ NAME="lr_0.02" ./scripts/prepare.sh python main.py -a resnet18 --lr 0.02
```
... or add it more permanently with `default_value SERIES "hyperparameter_optimization"` to the [default config](config/default).
