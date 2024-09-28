#!/usr/bin/env python3
"""
early_stopping module
"""


import numpy as np


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if gradient descent should stop early based on validation cost.

    Args:
        cost (float): Current validation cost of the neural network.
        opt_cost (float): Lowest recorded validation cost of
        the neural network.
        threshold (float): Threshold used for early stopping.
        patience (int): Patience count used for early stopping.
        count (int): Count of how long the threshold has not been met.

    Returns:
        bool: Whether to stop gradient descent early.
        int: Updated count of how long the threshold has not been met.
    """
    if cost < opt_cost - threshold:
        # Improvement in cost
        return False, 0
    else:
        # No significant improvement in cost
        count += 1
        if count >= patience:
            # Stop early if patience is exceeded
            return True, count
        else:
            return False, count
