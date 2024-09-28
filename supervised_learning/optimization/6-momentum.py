#!/usr/bin/env python3
"""
creates the training operation for a neural network in
tensorflow using the gradient descent with momentum
optimization algorithm:
"""

import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """
    Creates the training operation for a neural network in
    tensorflow using the gradient descent with momentum
    optimization algorithm.

    args:
        loss: the loss of the network
        alpha: the learning rate
        beta1: the momentum weight

    returns:
        the momentum optimization operation
    """
    optimizer = tf.train.MomentumOptimizer(alpha, beta1)
    return optimizer.minimize(loss)
