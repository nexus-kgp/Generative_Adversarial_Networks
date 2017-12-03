"""
models.py: Definitions for generator and discriminator convnets.
"""

import tensorflow as tf


# commonly used stride settings for 2D convolutions
STRIDE_1 = [1, 1, 1, 1]
STRIDE_2 = [1, 2, 2, 1]


def _bn(bottom, is_train):
    """
        Creates a batch normalization op.
        Meant to be invoked from other layer operations.
    """

    # assume inference by default
    if is_train is None:
        is_train = tf.constant(False, dtype="bool", name="is_train")

    # create scale and shift variables
    bn_shape = bottom.get_shape()[-1]
    shift = tf.get_variable("beta", shape=bn_shape,
                            initializer=tf.constant_initializer(0.0))
    scale = tf.get_variable("gamma", shape=bn_shape,
                            initializer=tf.constant_initializer(1.0))

    # compute mean and variance
    bn_axes = list(range(len(bottom.get_shape()) - 1))
    (mu, var) = tf.nn.moments(bottom, bn_axes)

    # batch normalization ops
    ema = tf.train.ExponentialMovingAverage(decay=0.9)

    def train_op():
        ema_op = ema.apply([mu, var])
        with tf.control_dependencies([ema_op]):
            return (tf.identity(mu), tf.identity(var))

    def test_op():
        moving_mu = ema.average(mu)
        moving_var = ema.average(var)
        return (moving_mu, moving_var)

    (mean, variance) = tf.cond(is_train, train_op, test_op)

    top = tf.nn.batch_normalization(bottom, mean, variance, shift, scale, 1e-4)

    return top


def conv2d(name, bottom, shape, strides, top_shape=None, with_bn=True, is_train=None):
    """
        Creates a convolution + BN block.
    """

    with tf.variable_scope(name) as scope:
        
        # apply batch normalization, if necessary
        bn = _bn(bottom, is_train) if with_bn else bottom

        # add convolution op
        weights = tf.get_variable("weights", shape=shape,
                                  initializer=tf.truncated_normal_initializer(stddev=0.02))
        if top_shape is not None:
            conv = tf.nn.conv2d_transpose(bn, weights, top_shape, strides, padding="SAME")
        else:
            conv = tf.nn.conv2d(bn, weights, strides, padding="SAME")

        # add biases
        bias_shape = [shape[-1]] if top_shape is None else [shape[-2]]
        biases = tf.get_variable("biases", shape=bias_shape,
                                 initializer=tf.constant_initializer())
        top = tf.nn.bias_add(conv, biases)

    return top
