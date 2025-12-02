import os

import pandas as pd

from .train_data_process import interpolate_data


# Define a function to load samples
def load_n_process(test_data):
    """
    Loads the sample
    Processes it
    Returns the sample
    in a format suitable
    for prediction
    """
    target_length = 55
    df = pd.read_csv(test_data, header=None, sep=',')
    # interpolate it
    interpolated_data = interpolate_data(df.to_numpy(), target_length)
    return interpolated_data
