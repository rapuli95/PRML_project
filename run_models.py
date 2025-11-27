import pickle
from NN_classifier.preprocessing import preprocess_sample
from NN_classifier.neural_network import NeuralNetworkClassifier
from SVM.svm_predictions import svm_predict
from statistics import mode

NN_model_path = "../trained_models/NN_classifier.pkl"
SVM_model_path = "../trained_models/SVM_classifier.pkl"
RF_model_path = "../trained_models/RF_classifier.pkl"


def run_model(self, data):
  with open(SVM_model_path, "rb") as f:
    svm_model = pickle.load(f)
    SVM_predictions = svm_predict(X, loaded_model)
  with open(NN_model_path, "rb") as f:
    nn_classifier_model: NeuralNetworkClassifier = pickle.load(f)
    X = preprocess_samples(data)
    NN_predictions = nn_classifier_model.predict(X)
  with open(SVM_model_path, "rb") as f:
    loaded_model = pickle.load(f)
    RF_predictions = nn_classifier_model.predict(X)
  # choose the most common prediction
  prediction = mode(SVM_predictions, NN_predictions, RF_predictions)
  return prediction
