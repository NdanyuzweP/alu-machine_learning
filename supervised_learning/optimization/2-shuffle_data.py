#!/usr/bin/env python3
"""
shuffles the data points in two matrices the same way
"""

import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way.

    Args:
        X (np.ndarray): The first matrix of shape (m, nx) where
        m is the number of data points and
        nx is the number of features.
        Y (np.ndarray): The second matrix of shape (m, ny) where
        m is the number of data points and
        ny is the number of features.

    Returns:
        The shuffled X and Y matrices.
    """
    m = X.shape[0]
    shuffle = np.random.permutation(m)
    return X[shuffle], Y[shuffle]
