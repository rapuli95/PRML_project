import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Define function to interpolate a sample
def interpolate_sample(sample: NDArray, target_length: int):
    
    # Cumulative arc length
    diffs = np.diff(sample, axis=0)
    segment_len = np.sqrt(np.sum(diffs**2, axis=1))
    cumlen = np.concatenate(([0], np.cumsum(segment_len)))
    total_len = cumlen[-1]
    
    # Create target archs
    target_arc_len = np.linspace(0, total_len, target_length)
    
    # Interpolate
    interpolated = np.zeros((target_length, sample.shape[1]))
    for coord  in range(sample.shape[1]):
        interpolated[:, coord] = np.interp(target_arc_len, cumlen, sample[:, coord])
    
    # Return interpolated sample
    return interpolated