# Efficient-Image-Super-Resolution

## SLURM 

#### Execute container
`srun -w fb10dl06 --gres gpu:1 --container-image /enroot_share/midl21t1/eisr.sqsh --container-mounts=/home/midl21t1/Efficient-Image-Super-Resolution:/home/midl21t1/Efficient-Image-Super-Resolution,/mnt/ssd1/midl21t1/datasets/:/mnt/ssd1/midl21t1/datasets/ --container-workdir=/home/midl21t1/Efficient-Image-Super-Resolution --container-writable --pty bash`


## Dependencies
* Python 3.6
* PyTorch >= 1.0.0
* numpy
* skimage
* **imageio**
* matplotlib
* tqdm
* cv2 >= 3.xx (Only if you want to use video input/output)

## Code
Clone this repository into any place you want.
```bash
git clone https://github.com/thstkdgus35/EDSR-PyTorch
cd EDSR-PyTorch
```

## Quickstart (Demo)
You can test our super-resolution algorithm with your images. Place your images in ``test`` folder. (like ``test/<your_image>``) We support **png** and **jpeg** files.

Run the script in ``src`` folder. Before you run the demo, please uncomment the appropriate line in ```demo.sh``` that you want to execute.
```bash
cd src       # You are now in */EDSR-PyTorch/src
sh demo.sh
```

You can find the result images from ```experiment/test/results``` folder.

| Model | Scale | File name (.pt) | Parameters | ****PSNR** |
|  ---  |  ---  | ---       | ---        | ---  |
| **EDSR** | 2 | EDSR_baseline_x2 | 1.37 M | 34.61 dB |
| | | *EDSR_x2 | 40.7 M | 35.03 dB |
| | 3 | EDSR_baseline_x3 | 1.55 M | 30.92 dB |
| | | *EDSR_x3 | 43.7 M | 31.26 dB |
| | 4 | EDSR_baseline_x4 | 1.52 M | 28.95 dB |
| | | *EDSR_x4 | 43.1 M | 29.25 dB |
| **MDSR** | 2 | MDSR_baseline | 3.23 M | 34.63 dB |
| | | *MDSR | 7.95 M| 34.92 dB |
| | 3 | MDSR_baseline | | 30.94 dB |
| | | *MDSR | | 31.22 dB |
| | 4 | MDSR_baseline | | 28.97 dB |
| | | *MDSR | | 29.24 dB |

*Baseline models are in ``experiment/model``. Please download our final models from [here](https://cv.snu.ac.kr/research/EDSR/model_pytorch.tar) (542MB)
**We measured PSNR using DIV2K 0801 ~ 0900, RGB channels, without self-ensemble. (scale + 2) pixels from the image boundary are ignored.

You can evaluate your models with widely-used benchmark datasets:

[Set5 - Bevilacqua et al. BMVC 2012](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html),

[Set14 - Zeyde et al. LNCS 2010](https://sites.google.com/site/romanzeyde/research-interests),

[B100 - Martin et al. ICCV 2001](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/),

[Urban100 - Huang et al. CVPR 2015](https://sites.google.com/site/jbhuang0604/publications/struct_sr).

For these datasets, we first convert the result images to YCbCr color space and evaluate PSNR on the Y channel only. You can download [benchmark datasets](https://cv.snu.ac.kr/research/EDSR/benchmark.tar) (250MB). Set ``--dir_data <where_benchmark_folder_located>`` to evaluate the EDSR and MDSR with the benchmarks.

You can download some results from [here](https://cv.snu.ac.kr/research/EDSR/result_image/edsr-results.tar).
The link contains **EDSR+_baseline_x4** and **EDSR+_x4**.
Otherwise, you can easily generate result images with ``demo.sh`` scripts.

## How to train EDSR and MDSR
We used [DIV2K](http://www.vision.ee.ethz.ch/%7Etimofter/publications/Agustsson-CVPRW-2017.pdf) dataset to train our model. Please download it from [here](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar) (7.1GB).

Unpack the tar file to any place you want. Then, change the ```dir_data``` argument in ```src/option.py``` to the place where DIV2K images are located.

We recommend you to pre-process the images before training. This step will decode all **png** files and save them as binaries. Use ``--ext sep_reset`` argument on your first run. You can skip the decoding part and use saved binaries with ``--ext sep`` argument.

If you have enough RAM (>= 32GB), you can use ``--ext bin`` argument to pack all DIV2K images in one binary file.

You can train EDSR and MDSR by yourself. All scripts are provided in the ``src/demo.sh``. Note that EDSR (x3, x4) requires pre-trained EDSR (x2). You can ignore this constraint by removing ```--pre_train <x2 model>``` argument.

```bash
cd src       # You are now in */EDSR-PyTorch/src
sh demo.sh
```
