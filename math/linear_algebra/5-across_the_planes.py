#!/usr/bin/env python3
"""
This module is implementing logic of adding matrices
"""


def add_matrices2D(mat1, mat2):
    """
    function add matrices that adds two matrices
    by adding two elements in the same position
    """
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    return [
        [mat1[i][j] + mat2[i][j] for j in range(len(mat1[0]))
         ] for i in range(len(mat1))]
