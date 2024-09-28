#!/usr/bin/env python3
"""
updates a variable using the gradient descent
with momentum optimization algorithm
"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using the gradient descent
    with momentun optimization algorithm.

    args:
        alpha: the learning rate
        beta1: the momentum weight
        var: np.ndarray containing the variable to be updated
        grad: np.ndarray containing the gradient of var
        v: the previous first moment of var

    returns:
        the updated variable and the new moment, respectively
    """
    v = beta1 * v + (1 - beta1) * grad
    var = var - alpha * v
    return var, v
