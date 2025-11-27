import pandas as pd
import os
from typing import List
from numpy.typing import NDArray
from sklearn.decomposition import PCA

dir_path = "../training_data"

class DataProcessing():
  def __init__(self, method):
    self.method = method
  def preprocess_train_data(self , path):
    """
    Upload the datasets
    and get features and the classes
    """
    names = []
    labels = [range(10)]
    datas = []
    datas_by_digit = {}
    for label in labels:
        if self.method == "SVM":
          key = str(label)
          datas_by_digit[key] = []
        for file_path in [os.path.join(dir_path, f) for f in os.listdir(dir_path) if ".csv" in f and f"stroke_{label}" in f]:
            df = pd.read_csv(file_path, header=None)
            datas.append(df)
            if self.method == "SVM":
              datas_by_digit[key].append(df)
    # return data
    if self.method == "SVM":
      returnable = [datas_by_digit, labels]
    else:
      returnable = [datas, labels]
    return {
        "labels": returnable[1],
        "datas": returnable[0],
    }
  def load_data(self,data_file):
    """
    Load and read a data file
    Returns the data
    """
    data = pd.read_csv(data_file, header=None)
    return data
  
  def project_to_2d(data: pd.DataFrame|NDArray, keep_order: bool=True, keep_dims: bool=False) -> NDArray:
    """
    Transform the data to
    two component PCA
    """
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

  def interpolate_data(points, target_length):
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

  def convert_to_image(data_projected: NDArray, size: tuple) -> NDArray: 
    """
    Convert the projected data
    to an image
    """
    # Initialize empty image
    width, height = size 
    image = np.zeros((height, width))

    # Center and scale the data to range [-1, 1]
    min_values, max_values = np.min(data_projected, axis=0), np.max(data_projected, axis=0)
    data_projected = data_projected - (min_values + max_values) / 2
    data_projected = data_projected / np.max(np.abs(data_projected)) # Scale to range [-1, 1]

    # Extract x and y data and scale to [0, N-1] (pixel coordinates)
    x_data, y_data = data_projected[:, 0], data_projected[:, 1] # Note: swap x and y for image coordinates
    x_data = x_data * ((width-1) / 2) + ((width-1) / 2)
    y_data = y_data * ((height-1) / 2) + ((height-1) / 2)

    # Compute floor and ceil values
    x_floor, x_ceil = np.floor(x_data).astype(int), np.ceil(x_data).astype(int)
    y_floor, y_ceil = np.floor(y_data).astype(int), np.ceil(y_data).astype(int)

    # Convert x, y data to pixels in an image
    image[y_floor, x_floor] += (1 - y_data % 1) * (1 - x_data % 1)
    image[y_ceil, x_ceil] += (y_data % 1) * (x_data % 1)
    image[y_floor, x_ceil] += (x_data % 1) * (1 - y_data % 1)
    image[y_ceil, x_floor] += (1 - x_data % 1) * (y_data % 1)

    # Clip the pixel values to range [0, 1]
    image = np.clip(image, 0, 1)  

    # Flip the image (so that it is not upside down)
    image = np.flip(image, 0)
    
    return image
