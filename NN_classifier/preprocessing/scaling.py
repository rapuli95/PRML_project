import numpy as np
from numpy.typing import NDArray

def min_max_scale(X: NDArray, axis: int=0) -> NDArray: 
    return (X - np.min(X, axis=axis)) / (np.max(X, axis=axis) - np.min(X, axis=axis))

def uniform_scale(X: NDArray, axis: int=0) -> NDArray: 
    return (X - np.mean(X, axis=axis)) / np.max(np.abs(X), axis=axis)