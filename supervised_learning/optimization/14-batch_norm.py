#!/usr/bin/env python3
"""
Creates a neural network with batch normalization in tensorflow.
"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in tensorflow.

    Args:
        prev: activated output of the previous layer
        n: number of nodes in the layer to be created
        activation: activation function that should be used on the output
        of the layer

    Returns:
        a tensor of activated output of the layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    x = tf.layers.Dense(units=n, kernel_initializer=init)(prev)
    mean, variance = tf.nn.moments(x, axes=[0])
    gamma = tf.Variable(tf.ones([n]))
    beta = tf.Variable(tf.zeros([n]))
    epsilon = 1e-8
    x_norm = tf.nn.batch_normalization(
        x, mean, variance, beta, gamma, epsilon
        )
    return activation(x_norm)
