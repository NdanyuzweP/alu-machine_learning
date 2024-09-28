#!/usr/bin/env python3
"""
calculates the normalization (standardization) constants of a matrix
"""
import numpy as np


def normalization_constants(X):
    """
    Calculates the normalization (standardization) constants of a matrix.

    Args:
        X (np.ndarray): The input data of shape (m, nx) where
        m is the number of data points and
        nx is the number of features.

    Returns:
        The mean and standard deviation of each feature, respectively.
    """
    return np.mean(X, axis=0), np.std(X, axis=0)
