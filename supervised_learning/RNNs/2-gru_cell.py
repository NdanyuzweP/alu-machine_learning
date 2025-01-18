#!/usr/bin/env python3
"""
This module contains the GRUCell class.
"""

import numpy as np


class BidirectionalCell:
    """
    This class represents a bidirectional cell of an RNN.
    """
    def __init__(self, i, h, o):
        """
        Constructor for the BidirectionalCell class.

        Args:
            i (int): Dimensionality of the data.
            h (int): Dimensionality of the hidden states.
            o (int): Dimensionality of the outputs.
        """
        # Weights and biases for the forward direction
        self.Whf = np.random.normal(size=(i + h, h))
        self.bhf = np.zeros((1, h))

        # Weights and biases for the backward direction
        self.Whb = np.random.normal(size=(i + h, h))
        self.bhb = np.zeros((1, h))

        # Weights and biases for the output
        self.Wy = np.random.normal(size=(2 * h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step in
        the forward direction.

        Args:
            h_prev (numpy.ndarray): Previous hidden state
            of shape (m, h).
            x_t (numpy.ndarray): Data input for the current time
            step of shape (m, i).

        Returns:
            h_next (numpy.ndarray): The next hidden state.
        """
        concat_input = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(concat_input, self.Whf) + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """
        Performs backward propagation for one time step in
        the backward direction.

        Args:
            h_next (numpy.ndarray): Next hidden state of shape (m, h).
            x_t (numpy.ndarray): Data input for the current time step
            of shape (m, i).

        Returns:
            h_prev (numpy.ndarray): The previous hidden state.
        """
        concat_input = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(np.dot(concat_input, self.Whb) + self.bhb)
        return h_prev

    def softmax(self, z):
        """
        Applies the softmax function to each element in z.

        Args:
            z (numpy.ndarray): Input array of shape (t, m, o).

        Returns:
            numpy.ndarray: Softmax-activated output of the same shape as z.
        """
        exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return exp_z / exp_z.sum(axis=-1, keepdims=True)

    def output(self, H):
        """
        Calculates all outputs for the RNN.

        Args:
            H (numpy.ndarray): Array of shape (t, m, 2 * h) that contains the
            concatenated hidden states from both directions, excluding their
            initialized states.

        Returns:
            Y (numpy.ndarray): The outputs of the RNN.
        """
        # Compute the linear transformation
        Y_linear = np.dot(H, self.Wy) + self.by
        
        # Apply softmax to get output probabilities
        Y = self.softmax(Y_linear)
        return Y
