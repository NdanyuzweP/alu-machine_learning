#!/usr/bin/env python3
'''
    Calculates the cost of a neural network with L2 regularization
'''


import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    ''' calculates the cost of a neural network with L2 regularization
        cost: float: cost of the network without L2 regularization
        lambtha: float: regularization parameter
        weights: dict: weights and biases of the network
        L: int: number of layers in the network
        m: int: number of data points used
        Returns: float: the cost of the network accounting for L2 reg
    '''
    W = 0
    for i in range(1, L + 1):
        W += np.linalg.norm(weights['W' + str(i)])
    return cost + lambtha / (2 * m) * W
