import os
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Optional, List
from .interpolate import interpolate_sample
from .project_to_2d import project_to_2d
from .convert_to_image import convert_to_image

DEFAULT_IMAGE_SIZE = (16, 16)

# Define a function to load samples
def load_samples(dir_path: str) -> dict: 
    labels, samples = [], []
    for filename, filepath in [(f, os.path.join(dir_path, f)) for f in os.listdir(dir_path) if ".csv" in f]: 
        labels.append(int(filename[7]))
        samples.append(pd.read_csv(filepath, header=None))
    return samples, labels

# Define a function to preprocess a single sample
def preprocess(sample: NDArray|pd.DataFrame, use_interpolation: bool=True, 
               use_PCA: bool=True, use_convert_to_image: bool=True, use_flatten: bool=True,
               target_len: Optional[int]=None, img_size: tuple=(16, 16)) -> NDArray:  
    assert use_interpolation or target_len is None, "Interpolation is enabled but target len was not provided"

    # Convert sample to numpy array if necessary
    if isinstance(sample, pd.DataFrame): sample = sample.to_numpy()

    # Interpolate sample (if target len was provided)
    if use_interpolation: sample = interpolate_sample(sample, target_len) 

    # Project sample to 2D (use PCA or take the first 2 dimensions)
    sample = project_to_2d(sample) if use_PCA else sample[:, :2]

    # Convert 2D coordinates to pixels in an image
    sample = convert_to_image(sample, img_size) if use_convert_to_image else sample

    # Return the image as a flattened vector 
    return sample.flatten() if use_flatten else sample

# Define function to preprocess multiple samples
def preprocess_samples(samples: List[NDArray|pd.DataFrame], img_size=DEFAULT_IMAGE_SIZE) -> NDArray: 
    target_len = max([len(s) for s in samples])
    X = np.array([preprocess(s, target_len=target_len, img_size=img_size) for s in samples])
    return X