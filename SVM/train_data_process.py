import os

import numpy as np
import pandas as pd

filepath = "training_data"

def interpolate_data(points, target_length):
  """
  Interpolates each dataset
  to a fixed number of points
  """
  #cumulative arc length
  diffs = np.diff(points, axis=0)
  segment_len = np.sqrt(np.sum(diffs**2, axis=1))
  cumlen = np.concatenate(([0], np.cumsum(segment_len)))
  total_len = cumlen[-1]
  # create target archs
  target_arc_len = np.linspace(0, total_len, target_length)
  # interpolate
  interpolated = np.zeros((target_length, points.shape[1]))
  for coord  in range(points.shape[1]):
    interpolated[:, coord] = np.interp(target_arc_len, cumlen, points[:, coord])
  # resample
  return interpolated

def load_n_interpolate_train_data():
  """
  Loads the training data
  Interpolates the data
  Returns a dictionary of interpolated data
  """
  str_numbers = list(range(10))
  str_filenums = [f"{i:04d}" for i in range(1,101)] # 0001 to 0100
  mean_lens = []
  # load the file names
  filefolder = os.listdir(filepath)
  files_in_order = sorted(filefolder)
  # go through each file and save the data
  file_data_dict = {}
  interpolated_dict = {}
  for i in str_numbers:
    key = str(i)
    file_data_dict[key] = []
    for j in str_filenums:
      filename = f"stroke_{i}_{j}.csv"
      df = pd.read_csv(f"{filepath}/{filename}",header=None, delimiter=',')
      df.columns = ['x', 'y','z']
      # interpolate the data
      file_data_dict[key].append(df)
      # once data loaded, determine target length
      # and interpolate
      num_rows = df.shape[0]
      # save the dataset means
      mean_lens.append(num_rows)
  # determine target length
  target_length = round(np.mean(mean_lens))
  # then interpolate each sample
  for j in str_numbers:
    key_new= str(j)
    interp_numbers = []
    for df in file_data_dict[key_new]:
      interpolated_data = interpolate_data(df.to_numpy(), target_length)
      interp_numbers.append(interpolated_data)
    # save the interpolated samples
    # to a dictionary ; key = digit
    interpolated_dict[key_new] = interp_numbers
  return interpolated_dict