import numpy as np
from sklearn.model_selection import train_test_split


def do_global_split(data_dict, test_size=0.2):
    """
    Takes all the samples
    Returns train test splits
    """
    X_all = []
    y_all = []
    filenames = []
    # add samples
    for digit in range(10):
        for sample_idx in range(100):
            sequence = data_dict[str(digit)][sample_idx]
            name = f"stroke_{digit}_{sample_idx+1:04d}"
            X_all.append(sequence)
            y_all.append(digit)
            filenames.append(name)
    # flatten the data
    X_flat = np.array(X_all).reshape(len(X_all),-1)
    y_arr = np.array(y_all)

    # split once globally
    X_train, X_test, y_train, y_test, filenames_train, filenames_test = train_test_split(
        X_flat, y_arr, filenames, test_size=test_size, random_state=42, stratify=y_arr
    )
    
    return X_train, X_test, y_train, y_test, filenames_test