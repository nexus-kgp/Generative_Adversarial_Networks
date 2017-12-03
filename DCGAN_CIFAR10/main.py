
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


def _clean_directory(path):
    """
        Clears (and creates) a directory on the filesystem.
    """

    if os.path.exists(path):
        shutil.rmtree(os.path.join(path))
    os.mkdir(path)





def _read_and_preprocess(paths, scale_len, crop_len):
    """
        Reads multiple images (and labels).
    """

    imgs = []

    for path in paths:
        img = imread(path)

        # force 3-channel images
        if img.ndim == 2:
            img = gray2rgb(img)
        elif img.shape[2] == 4:
            img = img[:, :, :3]

        # compute the resize dimension
        resize_f = float(scale_len) / min(img.shape[:2])
        new_dims = (int(np.round(img.shape[0] * resize_f)),
                    int(np.round(img.shape[1] * resize_f)))

        # prevent the input image from blowing up
        # factor of 2 is more or less an arbitrary number
        max_dim = 2 * scale_len
        new_dims = (min(new_dims[0], max_dim),
                    min(new_dims[1], max_dim))

        # resize and center crop
        img = resize(img, new_dims)
        top = int(np.ceil((img.shape[0] - crop_len) / 2.0))
        left = int(np.ceil((img.shape[1] - crop_len) / 2.0))
        img = img[top:(top+crop_len), left:(left+crop_len)]

        # preprocessing (tanh)
        img = 2 * img - 1

        imgs.append(img)

    return np.array(imgs)


def _deprocess_and_save(batch_res, epoch, grid_shape=(8, 8), grid_pad=5):
    """
        Deprocesses the generator output and saves the results.
    """

    # create an output grid to hold the images
    (img_h, img_w) = batch_res.shape[1:3]
    grid_h = img_h * grid_shape[0] + grid_pad * (grid_shape[0] - 1)
    grid_w = img_w * grid_shape[1] + grid_pad * (grid_shape[1] - 1)
    img_grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    # loop through all generator outputs
    for i, res in enumerate(batch_res):
        if i >= grid_shape[0] * grid_shape[1]:
            break

        # deprocessing (tanh)
        img = (res + 1) * 127.5
        img = img.astype(np.uint8)

        # add the image to the image grid
        row = (i // grid_shape[0]) * (img_h + grid_pad)
        col = (i % grid_shape[1]) * (img_w + grid_pad)

        img_grid[row:row+img_h, col:col+img_w, :] = img

    # save the output image
    fname = "epoch{0}.jpg".format(epoch) if epoch >= 0 else "result.jpg"
    imsave(os.path.join(OUTPUT_PATH, fname), img_grid)
