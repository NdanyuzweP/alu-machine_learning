#!/usr/bin/env python3
"""
Dropout Forward Propagation
"""


import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Function that conducts forward propagation using Dropout
    Arguments:
     - X is a numpy.ndarray of shape (nx, m) containing
     the input data for the network
     - weights is a dictionary of the weights and biases
     of the neural network
     - L the number of layers in the network
     - keep_prob is the probability that a node will be kept

    Returns:
    - a dictionary containing the outputs of each layer and
    the dropout mask used on each layer
    """
    cache = {}
    cache['A0'] = X
    for i in range(L):
        W = weights['W' + str(i + 1)]
        b = weights['b' + str(i + 1)]
        A = cache['A' + str(i)]
        Z = np.matmul(W, A) + b

        if i == L - 1:
            # softmax activation function
            t = np.exp(Z)
            cache['A' + str(i + 1)] = t / np.sum(
                t, axis=0, keepdims=True
                )
        else:
            # tanh activation function
            A = np.tanh(Z)
            D = np.random.rand(A.shape[0], A.shape[1])
            D = np.where(D < keep_prob, 1, 0)
            A = A * D
            A = A / keep_prob
            cache['D' + str(i + 1)] = D
            cache['A' + str(i + 1)] = A

    return cache
