#!/usr/bin/env python3
""" Privatizing neuron class """
import numpy as np


class Neuron:
    """ Class that defines a single neuron performing binary classification """

    def __init__(self, nx):
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ W getter """
        return self.__W

    @property
    def b(self):
        """ b getter """
        return self.__b

    @property
    def A(self):
        """ A getter """
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron
        X: is a numpy.ndarray with shape (nx, m) that contains the input data
        nx: is the number of input features to the neuron
        m: is the number of examples
        The neuron should use a sigmoid activation function
        Returns the private attribute __A
        """
        self.__A = 1 / (1 + np.exp(-(np.matmul(self.__W, X) + self.__b)))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        Y is a numpy.ndarray with shape (1, m) that contains the correct labels
        for the input data
        A is a numpy.ndarray with shape (1, m) containing the activated output
        of the neuron for each example
        To avoid division by zero errors, use 1.0000001 - A instead of 1 - A
        Returns the cost
        """
        m = Y.shape[1]
        return -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
