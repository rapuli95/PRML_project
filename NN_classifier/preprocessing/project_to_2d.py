import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.decomposition import PCA

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