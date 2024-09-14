#!/usr/bin/env python3

"""
simplest neural network with no hidden layer

"""

import numpy as np


class Neuron:
    """
    Class that defines a single neuron performing binary classification
    initialization:
        nx: number of input features to the neuron
        If nx is not an integer, raise a TypeError: nx must be an integer
        If nx is less than 1, raise a ValueError nx must be a positive integer
        All exceptions should be raised in the order listed
        The weights vector W and bias b for the neuron are
        initialized using random normal distribution
            The bias b is initialized to 0
    """

    def __init__(self, nx):
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')

        self.nx = nx
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
