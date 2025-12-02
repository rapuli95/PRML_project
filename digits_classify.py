import os
import sys
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from numpy.typing import NDArray
from pandas import DataFrame

# IMPORTANT: add implementations to path
sys.path.append("NN_classifier")
sys.path.append("Random_forest")

# Import neural network implementation
from NN_classifier.neural_network import NeuralNetworkClassifier
from NN_classifier.NN_preprocessing import NN_preprocess_samples as NN_preprocessing, load_samples

# Import random forest implementation
from Random_forest.RandomForest import RandomForest
from Random_forest.RF_preprocessing import preprocess_samples as RF_preprocessing

# Define default model
DEFAULT_MODEL = os.path.join("trained_models", "NN_classifier_d5_p1186826_e2000_98_20251202_0112_deeplearning.pkl")

# Define function for classifying 3D digits
def digits_classify(test_data: NDArray|list[NDArray|DataFrame], model_path: str=DEFAULT_MODEL) -> NDArray: 
    
    # Load trained model
    try: 
        with open(model_path, "rb") as f: 
            model = pickle.load(f)
    except FileNotFoundError: 
        print(f"Could not load model from {model_path}")
        return None
    
    # Choose preprocessing pipeline according to model type
    if "NeuralNetworkClassifier" in str(type(model)): 
        preprocess_samples = NN_preprocessing
    elif "RandomForest" in str(type(model)): 
        preprocess_samples = RF_preprocessing
    else: 
        raise NotImplementedError
    
    # Preprocess samples
    preprocessed_data = preprocess_samples(test_data)

    # Classify preprocessed samples
    predicted_classes = model.predict(preprocessed_data)

    # Return predicted classes
    return predicted_classes

# Test classification function
if __name__ == "__main__": 

    # Load the data
    samples, labels = load_samples("training_data")
    labels = np.array(labels)

    # Extract a random test set
    _, test_data, _, test_labels = train_test_split(samples, labels, test_size=0.2, stratify=labels)

    # Predict the samples
    predictions = digits_classify(test_data)

    # Evaluate prediction accuracy
    print(f"Prediction accuracy: {accuracy_score(predictions, test_labels)*100:.1f} %")
    print(classification_report(predictions, test_labels))