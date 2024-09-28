#!/usr/bin/env python3
"""
Creates a layer that includes L2 regularization
usinf TensorFlow
"""


import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a TensorFlow layer that includes L2 regularization.

    Parameters:
    prev -- tensor containing the output of the previous layer
    n -- number of nodes the new layer should contain
    activation -- activation function that should be used on the layer
    lambtha -- L2 regularization parameter

    Returns:
    The output of the new layer
    """
    kernel = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    reg = tf.contrib.layers.l2_regularizer(lambtha)
    layer = tf.layers.Dense(n, activation=activation,
                            kernel_initializer=kernel,
                            kernel_regularizer=reg)
    return layer(prev)
