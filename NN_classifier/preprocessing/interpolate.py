import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Define function to interpolate a sample
def interpolate_sample(sample: NDArray, target_length: int):
    if isinstance(sample, pd.DataFrame): sample = sample.to_numpy()
    if len(sample) == target_length: return sample
    
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

# Define function to interpolate a sample
def resample(sample: NDArray, target_length: int):
    assert target_length > 0, "Target length has to be greater than 0"
    if isinstance(sample, pd.DataFrame): sample = sample.to_numpy()

    # Check if sample is longer or shorter than target length
    if len(sample) > target_length: 

        # Return a randomly selected subset of the sample
        index = np.sort(np.random.choice(len(sample), target_length))
        return sample[index]
    
    else: 

        # Interpolate new datapoints
        return interpolate_sample(sample, target_length)
    
# Define function to interpolate a sample
def resample2(sample: NDArray, target_length: int):
    assert target_length > 0, "Target length has to be greater than 0"
    if isinstance(sample, pd.DataFrame): sample = sample.to_numpy()

    # Interpolate sample to double the target length
    sample = interpolate_sample(sample, target_length * 2)

    # Return a resampled sample with every other data point
    return sample[0::2]