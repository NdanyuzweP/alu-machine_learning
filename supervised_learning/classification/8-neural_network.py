#!/usr/bin/env python3

"""
neural network with minimum number of layers being
2 nwurons and 1 hidden layer

"""

import numpy as np


class NeuralNetwork:
    """
    Class that defines a neural network with one hidden layer
    performing binary classification
    """

    def __init__(self, nx, nodes):
        """
        nx is the number of input features to the neuron
        nodes is the number of nodes found in the hidden layer
        Private instance attributes:
        __W1: The weights vector for the hidden layer. Upon instantiation,
        it should be initialized using a random normal distribution.
        __b1: The bias for the hidden layer. Upon instantiation,
        it should be initialized with 0s.
        __A1: The activated output for the hidden layer. Upon instantiation,
        it should be initialized to 0.
        __W2: The weights vector for the neuron. Upon instantiation,
        it should be initialized using a random normal distribution.
        __b2: The bias for the neuron. Upon instantiation,
        it should be initialized to 0.
        __A2: The activated output for the neuron. Upon instantiation,
        it should be initialized to 0.
        """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')

        if not isinstance(nodes, int):
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')

        self.W1 = np.random.randn(nodes, nx)
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0

        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0
