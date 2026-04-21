import numpy as np
from numba import njit


@njit
def _partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


@njit
def _quicksort(arr, low, high):
    if low < high:
        pivot_index = _partition(arr, low, high)
        _quicksort(arr, low, pivot_index - 1)
        _quicksort(arr, pivot_index + 1, high)


@njit
def quicksort(arr: np.array) -> np.array:
    arr = arr.copy()
    _quicksort(arr, 0, len(arr) - 1)
    return arr
