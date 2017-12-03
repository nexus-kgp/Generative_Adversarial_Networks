
# system imports
import argparse
import glob
import os
import random
import shutil

# library imports
import numpy as np
from skimage.color import gray2rgb
from skimage.io import imread, imsave
from skimage.transform import resize
import tensorflow as tf

# local imports
from models import generator, discriminator


# size of the input latent space
Z_SIZE = 64

# training parameters
TRAIN_RATIO = 10
DISPLAY_LOSSES = 50

# directory to snapshot models and images
OUTPUT_PATH = "outputs"
CHECKPOINT_NAME = "dcgan.tfmodel"


# argparse
parser = argparse.ArgumentParser(description="Train a DCGAN using Tensorflow.")
parser.add_argument("-n", "--num-epochs", type=int, default=100, help="number of epochs")
parser.add_argument("-b", "--batch-size", type=int, default=128, help="batch size to use")
parser.add_argument("-l", "--learning-rate", type=float, default=1e-3, help="generator learning rate")
parser.add_argument("-i", "--image-size", type=int, default=64, help="(square) image size")
parser.add_argument("-s", "--scale-size", type=int, default=64, help="resize length for center crop")
parser.add_argument("-t", "--train-dir", type=str, help="directory to pull training images from")
parser.add_argument("-o", "--output-dir", type=str, default="outputs", help="directory to output generations")
parser.add_argument("-r", "--restore", action="store_true", help="specify to use the latest checkpoint")

