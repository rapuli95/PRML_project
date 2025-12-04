import os
import sys
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from numpy.typing import NDArray
from pandas import DataFrame

# Import neural network implementation
from NN_classifier.neural_network import NeuralNetworkClassifier
from NN_classifier.preprocessing import NN_preprocess_samples as NN_preprocessing, load_samples

# Import random forest implementation
from Random_forest.RandomForest import RandomForest
from Random_forest.RF_preprocessing import preprocess_samples as RF_preprocessing

# Import support vector machine implementation
from SVM.train_data_process import interpolate_data
from SVM.svm_setup import svm_predict

# Define default model
DEFAULT_MODEL = os.path.join("trained_models", "NN_classifier.pkl")

# Define function for classifying 3D digits
def digits_classify(test_data: NDArray|list[NDArray|DataFrame], model_path: str=DEFAULT_MODEL) -> NDArray: 
    
    # Load trained model
    try: 
        with open(model_path, "rb") as f: 
            model = pickle.load(f)
    except FileNotFoundError: 
        print(f"Could not load model from {model_path}")
        return None
    
    # Convert samples to numpy arrays
    test_data = [s.to_numpy() if isinstance(s, DataFrame) else s for s in test_data]
    
    # Choose preprocessing pipeline according to model type
    if isinstance(model, NeuralNetworkClassifier): 
        
        # Preprocess data
        preprocessed_data = NN_preprocessing(test_data)

        # Classify preprocessed samples
        predicted_classes = model.predict(preprocessed_data)
    
    elif isinstance(model, RandomForest): 

        # Preprocess data
        preprocessed_data = RF_preprocessing(test_data)

        # Classify preprocessed samples
        predicted_classes = model.predict(preprocessed_data)

    elif "svm" in model_path: 

        # Preprocess data
        preprocessed_data = [interpolate_data(s, 55).flatten() for s in test_data]

        # Classify preprocessed samples
        predicted_classes = [svm_predict(s, model_path) for s in preprocessed_data]

    else: 
        raise NotImplementedError

    # Return predicted classes 
    return predicted_classes if len(predicted_classes) > 1 else int(predicted_classes[0])

# Test classification function
if __name__ == "__main__": 

    # Load the data
    samples, labels = load_samples("training_data")
    labels = np.array(labels)

    # Extract a random test set
    _, test_data, _, test_labels = train_test_split(samples, labels, test_size=0.2, stratify=labels)

    # Predict the samples
    predictions = digits_classify(test_data)
    # predictions = digits_classify(test_data, os.path.join("trained_models", "RF_classifier.pkl"))
    # predictions = digits_classify(test_data, os.path.join("trained_models", "svm_classifier.pkl"))

    # Evaluate prediction accuracy
    print(f"Prediction accuracy: {accuracy_score(predictions, test_labels)*100:.1f} %")
    print(classification_report(predictions, test_labels))