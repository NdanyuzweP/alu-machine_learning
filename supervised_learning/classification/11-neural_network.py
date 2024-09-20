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

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """
        returns private instance attribute __W1
        """
        return self.__W1

    @property
    def b1(self):
        """
        returns private instance attribute __b1
        """
        return self.__b1

    @property
    def A1(self):
        """
        returns private instance attribute __A1
        """
        return self.__A1

    @property
    def W2(self):
        """
        returns private instance attribute __W2
        """
        return self.__W2

    @property
    def b2(self):
        """
        returns private instance attribute __b2
        """
        return self.__b2

    @property
    def A2(self):
        """
        returns private instance attribute __A2
        """
        return self.__A2

    def forward_prop(self, X):
        """
        calculates the forward propagation of the neural network
        X is a numpy.ndarray with shape (nx, m) that contains the input data
        nx is the number of input features to the neuron
        m is the number of examples
        Updates the private attributes __A1 and __A2
        The neurons should use a sigmoid activation function
        Returns the private attributes __A1 and __A2, respectively
        """
        self.__A1 = 1 / (1 + np.exp(-(np.matmul(self.__W1, X) + self.__b1)))
        self.__A2 = 1 / \
            (1 + np.exp(-(np.matmul(self.__W2, self.__A1) + self.__b2)))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        calculates the cost of the model using logistic regression
        Y is a numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data
        A is a numpy.ndarray with shape (1, m) containing the activated
        output of the neuron for each example
        To avoid division by zero errors, use 1.0000001 - A instead of 1 - A
        Returns the cost
        """
        m = Y.shape[1]
        return -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
