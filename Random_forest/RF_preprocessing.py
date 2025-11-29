from sklearn.decomposition import PCA
from numpy.typing import NDArray
from typing import Optional, List
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Define a function to load samples
def load_samples(dir_path: str) -> dict: 
    labels, samples = [], []
    for filename, filepath in [(f, os.path.join(dir_path, f)) for f in os.listdir(dir_path) if ".csv" in f]: 
        labels.append(int(filename[7]))
        samples.append(pd.read_csv(filepath, header=None))
    return samples, labels



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
        index = np.random.choice(len(sample), target_length).sort()
        return sample[index]
    
    else: 

        # Interpolate new datapoints
        return interpolate_sample(sample, target_length)



# Define function to project 3D data to 2D using PCA
def project_to_2d(data: pd.DataFrame|NDArray, keep_order: bool=True, keep_dims: bool=False) -> NDArray:
    
    # Convert data to numpy array if needed
    if isinstance(data, pd.DataFrame): data = data.to_numpy()

    # Apply PCA
    pca = PCA(n_components=data.shape[1]).fit(data)
    scores = pca.fit_transform(data)
    loadings = pca.components_ 

    # Find principal components that align best with the original axes
    if keep_order: 
        order = np.argmax(np.abs(loadings), axis=0)[:(data.shape[1] if keep_dims else 2)]
    else: 
        order = np.arange(data.shape[1]) if keep_dims else np.arange(2)

    # Return reordered projected data
    return scores[:, order]



# Define a function to preprocess a single sample
def preprocess(sample: NDArray|pd.DataFrame, use_interpolation: bool=True, 
               use_PCA: bool=True, use_flatten: bool=True,
               target_len: Optional[int]=None):
    # If interpolation is enabled, target_len must be provided
    assert (not use_interpolation) or (target_len is not None), "Interpolation is enabled but target len was not provided"
    
    # Convert sample to numpy array if necessary
    if isinstance(sample, pd.DataFrame):
        sample = sample.to_numpy()

    # Interpolate sample (if target len was provided)
    if use_interpolation:
        sample = resample(sample, target_len)

    # Project sample to 2D (use PCA or take the first 2 dimensions)
    sample = project_to_2d(sample) if use_PCA else sample[:, :2]

    return sample.flatten() if use_flatten else sample



# Define function to preprocess multiple samples
def preprocess_samples(samples: List[NDArray|pd.DataFrame], target_len: int=256) -> NDArray: 
    assert target_len > 0, "Target length must be greater than 0"
    X = np.array([preprocess(s, target_len=target_len) for s in samples])
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X
