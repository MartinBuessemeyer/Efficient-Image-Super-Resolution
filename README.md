# Efficient Image Super Resolution

This repository conatains the source for our Machine Intelligence with Deep Learning (MIDL) seminar topic _Efficient Image Super Resolution_ from the winter term 2021/2022.

The project goal is to extend the already existing the [Residual Feature Distillation Network](https://arxiv.org/abs/2009.11551) to increase speed and accuracy while simultaneously decreasing model size.  

## Set Up

### General

1. Clone the repository.
```bash
git clone git@github.com:MartinBuessemeyer/Efficient-Image-Super-Resolution.git
```

2. Get the data sets
    - DIV2K
    - Set5
    - Set14 
    - BSD100
    - Urban100

3. Build the enroot container. This will automatically handle all dependencies for you.
```bash
sh ./scripts/build-image-enroot.sh
```

## How to execute

1. Run the container. The following steps should be exuted inside the enroot continer.

2. Adjust the `src/run.sh`. You can find all the available options in the `src/options.py`. 

3. Run the `src/run.sh`.
```bash
sh ./src/run.sh
```

## Authors

Martin Büßemeyer, Björn Daase, and Maximilian Kleissl

## License

```
# SPDX-License-Identifier: MIT
```