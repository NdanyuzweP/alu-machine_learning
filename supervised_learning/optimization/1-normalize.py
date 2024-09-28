#!/usr/bin/env python3
"""
normalizes (standardizes) a matrix
"""

import numpy as np


def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix.

    Args:
        X (np.ndarray): The input data of shape (m, nx) where
        m is the number of data points and
        nx is the number of features.
        m (np.ndarray): The mean of all the features of X.
        s (np.ndarray): The standard deviation of all the features of X.

    Returns:
        The normalized X matrix.
    """
    return (X - m) / s
