import numpy as np
from numpy.typing import NDArray

def one_hot_encode(y: NDArray, classes: NDArray=None) -> NDArray: 
    assert len(y) == y.size, 'y must be a vector'
    assert classes is None or len(classes) == classes.size, 'classes must be a vector'
    if classes is None: classes = np.arange(np.max(y))
    y_encoded = np.zeros((len(y), len(classes)))
    y_encoded[np.arange(len(y)), y] = 1
    return y_encoded 