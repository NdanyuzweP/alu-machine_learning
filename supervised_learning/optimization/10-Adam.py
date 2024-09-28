#!/usr/bin/env python3
"""
creates the training operation for a neural network
in tensorflow using the Adam optimization algorithm.
"""
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    creates the training operation for a neural network
    in tf using the Adam op.

    args:
        loss: the loss of the network
        alpha: the learning rate
        beta1: the weight used for the first moment
        beta2: the weight used for the second moment
        epsilon: a small number to avoid division by zero

    returns:
        the Adam optimization operation
    """
    optimizer = tf.train.AdamOptimizer(
        learning_rate=alpha, beta1=beta1, beta2=beta2, epsilon=epsilon
        )
    train_op = optimizer.minimize(loss)
    return train_op
