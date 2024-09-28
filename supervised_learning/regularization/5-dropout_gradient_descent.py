#!/usr/bin/env python3
"""
 Updates the weights of a neural network with
 Dropout regularization using gradient descent.
"""


import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout regularization
    using gradient descent.

    Parameters:
    Y -- numpy.ndarray of shape (classes, m) with correct labels for the data
    weights -- dictionary of the weights and biases of the neural network
    cache -- dictionary of the outputs and dropout masks of
    each layer of the neural network
    alpha -- learning rate
    keep_prob -- probability that a node will be kept
    L -- number of layers of the network

    Returns:
    Updates the weights of the network in place
    """
    m = Y.shape[1]
    dz = cache["A" + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache["A" + str(i - 1)]
        W = weights["W" + str(i)]
        b = weights["b" + str(i)]

        dw = (1 / m) * np.matmul(dz, A_prev.T)
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

        if i > 1:
            D_prev = cache["D" + str(i - 1)]
            dz = np.matmul(W.T, dz) * (1 - A_prev ** 2)
            dz = dz * D_prev / keep_prob  # Apply dropout mask

        # Update weights and biases
        weights["W" + str(i)] -= alpha * dw
        weights["b" + str(i)] -= alpha * db
