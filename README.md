# GAN Cocktail: mixing GANs without dataset access

<a href="https://arxiv.org/abs/2106.03847"><img src="https://img.shields.io/badge/arXiv-2106.03847-b31b1b.svg"></a>
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch->=1.7.1-Red?logo=pytorch"></a>

> **GAN Cocktail: mixing GANs without dataset access**
>
> Omri Avrahami, Dani Lischinski, Ohad Fried
>
> Abstract: Today's generative models are capable of synthesizing high-fidelity images, but each model specializes on a specific target domain. This raises the need for model merging: combining two or more pretrained generative models into a single unified one. In this work we tackle the problem of model merging, given two constraints that often come up in the real world: (1) no access to the original training data, and (2) without increasing the network size. To the best of our knowledge, model merging under these constraints has not been studied thus far. 
We propose a novel, two-stage solution. In the first stage, we transform the weights of all the models to the same parameter space by a technique we term model rooting. In the second stage, we merge the rooted models by averaging their weights and fine-tuning them for each specific domain, using only data generated by the original trained models. We demonstrate that our approach is superior to baseline methods and to existing transfer learning techniques, and investigate several applications.

## Requirements
The same requirements from [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch):
* 1&ndash;8 high-end NVIDIA GPUs with at least 12 GB of memory.
* 64-bit Python 3.7 and PyTorch 1.7.1. See [https://pytorch.org/](https://pytorch.org/) for PyTorch install instructions.
* CUDA toolkit 11.0 or later.  Use at least version 11.1 if running on RTX 3090.

## Getting started
### Creating a virtual environment
Create a new virtual environment using conda:
```.bash
$ conda env create -f environment.yml
$ conda activate gan_cocktail
```

Datasets are stored as uncompressed ZIP archives containing uncompressed PNG files and a metadata file `dataset.json` for labels.

### Preparing the datasets
Custom datasets can be created from a folder containing images; see `python dataset_tool.py --help` for more information.

**FFHQ**:

Step 1: Download the [Flickr-Faces-HQ dataset](https://github.com/NVlabs/ffhq-dataset) as TFRecords.

Step 2: Extract images from TFRecords using `dataset_tool.py` from the [TensorFlow version of StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada/):

```.bash
# Using dataset_tool.py from TensorFlow version at
# https://github.com/NVlabs/stylegan2-ada/
$ python ../stylegan2-ada/dataset_tool.py unpack \
    --tfrecord_dir=~/ffhq-dataset/tfrecords/ffhq --output_dir=/tmp/ffhq-unpacked
```

Step 3: Create ZIP archive using `dataset_tool.py` from this repository:

```.bash
# Scaled down 128x128 resolution.
$ python dataset_tool.py --source=/tmp/ffhq-unpacked --dest=~/datasets/ffhq_128.zip --width=128 --height=128
```

**LSUN**: Download the desired categories from the [LSUN project page](https://www.yf.io/p/lsun/) and convert to ZIP archive:

```.bash
$ python dataset_tool.py --source=~/downloads/lsun/raw/cat_lmdb --dest=~/datasets/lsuncat200k.zip --transform=center-crop --width=128 --height=128 --max_images=100000
```

### Train
In order to pretrain the source models you should run:
```.bash
python train.py --data_paths PATH_TO_DATASET --merge_mode False
```

Next, in order to perform the GAN cocktail merging you need to follow these 2 stages:

1. Model rooting
```.bash
$ python train.py --data_paths PATH_TO_DATASET1 --data_paths PATH_TO_DATASET2 --merge_model_paths PATH_TO_MODEL1 --merge_model_paths PATH_TO_MODEL2 --class_percentages 0 --class_percentages 1
```

2. Model merging
```.bash
$ python train.py --data_paths PATH_TO_DATASET1 --data_paths PATH_TO_DATASET2 --merge_model_paths PATH_TO_MODEL1 --merge_model_paths PATH_TO_ROOTED_MODEL2
```

### Visualize results
Generating images from the trained model:
```.bash
$ python generate.py --outdir=output/generate --seeds=1,2,4 --network=MODEL_PATH
```

Interpolate between samples of the merged model:
```.bash
$ python interpolate_samples.py --outdir=output/interpolate --seeds0=200 --seed1=100 --network=MODEL_PATH
```

Style mixing example:
```.bash
$ python style_mixing.py --outdir=output/style_mixing --rows=1,2,3 --cols=10,11,12 --network=MODEL_PATH
```

## Acknowledgements
Based on the Pytorch implementation of [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch).

## Citation
If you find this project useful, please cite the following:
```
@article{avrahami2021gan,
  title={GAN Cocktail: mixing GANs without dataset access},
  author={Avrahami, Omri and Lischinski, Dani and Fried, Ohad},
  journal={arXiv preprint arXiv:2106.03847},
  year={2021}
}
```