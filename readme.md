## Introduction
This is a tensorflow implementation of [pointnet](https://arxiv.org/abs/1612.00593) based on the [official implementation](https://github.com/charlesq34/pointnet). Main difference is the thorough replacement of conv2d operations by tf.layers.dense.

## Usage
To train the classification model:

```
python train.py
```
ModelNet40 models in HDF5 files will be downloaded to `./data`. Training results, including model parameter, hyperparameter setting and log files, will be saved into `./training_results/<training time>`.

You can tune hyperparameters in the `base_config.ini` file. Each time you train the model, this file will be read and copied into `./training_results/<training time>`.
