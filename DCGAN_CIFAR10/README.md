# tf-dcgan

## Introduction

This repository contains a TensorFlow implementation of [Deep Convolutional Generative Adversarial Networks](http://arxiv.org/abs/1511.06434), with some modifications. Specifically, this implementation makes the following changes:

 - An extra 3x3 convolutional layer is added to the end of the generator network. The body of the discriminator network is left unchanged.
 - Batch normalization is applied to the input of each convolutional layer instead of after.
 - Global average pooling is utilized in the discriminator instead of a densely connected layer.

The discriminator uses ReLU non-linearities instead of LReLU. 

These changes were made for experimental purposes, and seem to improve both stability and visual results datasets I'm interested in. They may or may not work better for others. For a DCGAN implementation with the models used in the paper, switch to the `original` branch.

## Requirements

 - Python (2.7 or 3.5)
 - TensorFlow >= 0.8

## Usage

To train your own model, clone this repository and run:

```
python dcgan.py -t <train_dir> -o <output_dir>
```

This will snapshot weights as well as intermediate generations to the directory specified by `<output_dir>`. To generate samples from a trained model, simply omit the `-t` flag:

```
python dcgan.py -o <output_directory>
```

For more options, run:

```
python dcgan.py -h
```
