#!/usr/bin/env python3

"""
 arr = [9, 8, 2, 3, 9, 4, 1, 0, 3]
 this module will return the size of the array
"""


def matrix_shape(matrix):
    """
    matrix: list of list
    return: list of integers
    """
    shape = []
    while type(matrix) == list:
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape


if __name__ == "__main__":
    matrix_shape_function = __import__('2-size_me_please').matrix_shape
