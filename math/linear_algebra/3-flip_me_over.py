#!/usr/bin/env python3

"""
this module will return the transpose of a matrix
this means that the rows will become columns and the columns will become rows
"""


def matrix_transpose(matrix):
    """
    matrix: list of list
    return: list of list
    """
    transpose = []
    for i in range(len(matrix[0])):
        row = []
        for j in range(len(matrix)):
            row.append(matrix[j][i])
        transpose.append(row)
    return transpose
