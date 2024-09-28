#!/usr/bin/env python3
"""
Function that calculates the cost of a neural network
with L2 regularization
using Tensorflow
"""


import tensorflow as tf


def l2_reg_cost(cost):
    """
    Function that calculates the cost of a neural network
    with L2 regularization
    Arguments:
     - cost is a tensor containing the cost of the network
    without L2 regularization
    Returns:
     A tensor containing the cost of the network
     accounting for L2 regularization
    """
    return cost + tf.losses.get_regularization_losses()
