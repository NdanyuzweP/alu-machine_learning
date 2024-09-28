#!/usr/bin/env python3
"""
calculates the weighted moving average of a data set
"""
import numpy as np


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set.

    Args:
        data (np.ndarray): The data to calculate the moving average of.
        beta (float): The weight used for the moving average.

    Returns:
        np.ndarray: The moving averages of data.
    """
    v = 0
    moving_averages = []
    for i in range(len(data)):
        v = beta * v + (1 - beta) * data[i]
        moving_averages.append(v / (1 - beta ** (i + 1)))
    return moving_averages
