#!/usr/bin/env python3
"""
This module is implementing logic of adding two array
it has to add two element in the same  position
itn the array
"""


def add_arrays(arr1, arr2):
    """
function add arrays that adds two arrays
by adding two elements in the same position
    """
    if len(arr1) != len(arr2):
        return None

    result = []
    for num in range(len(arr1)):
        result.append(arr1[num] + arr2[num])
    return result
