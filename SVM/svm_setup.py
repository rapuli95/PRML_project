import os
import pickle
import random
from itertools import combinations

import numpy as np
from cvxopt import matrix, solvers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def svm_linear_fit(data, class_labels):
  N = data.shape[0] # number of samples
  LB = 0
  C = 1.0
  # let's define P q g H A b
  K = (data @ data.T)
  reg_factor = 1e-6 * np.trace(K) / N  # Adaptive regularization
  P_matrix = (np.outer(class_labels, class_labels) * K + reg_factor * np.eye(N))
  P = matrix(P_matrix.astype(float))
  q = matrix(-np.ones(N)) # correct
  G = matrix(np.vstack((-np.eye(N), np.eye(N)))) # correct
  h = matrix(np.hstack((np.zeros(N), C*np.ones(N)))) # correct
  A = matrix(class_labels.reshape(1, -1), tc='d') # correct
  b = matrix(0.0) # correct
  # solve the quadratic problem
  solvers.options['show_progress'] = False
  sol = solvers.qp(P, q, G, h, A, b)
  # get the lambdas
  lambdas = np.array(sol['x']).flatten()
  # get the support vectors
  tol = 1e-5
  supp_sv_idx = (lambdas > tol)
  sum_sp_idx = np.sum(supp_sv_idx)
  supp_vc = data[supp_sv_idx]
  supp_labels = class_labels[supp_sv_idx]
  # get the weights
  w = np.sum((lambdas * class_labels)[:,None]* data, axis=0)
  w0_value = supp_labels - np.dot(supp_vc,w)
  w0 = np.mean(w0_value)
  return [w, w0]

def train_svm_model(digit1, digit2, X_train, y_train):
    """
    Trains the combinations
    Returns the weights
    """
    mask = (y_train == digit1) | (y_train == digit2)
    X_pair = X_train[mask]
    y_pair = y_train[mask]
    # convert to binary labels
    y_binary = np.where(y_pair == digit1, -1, 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pair)
    # get weights
    weights = svm_linear_fit(X_scaled, y_binary)
    return weights, scaler

def train_and_save_svm_classifier(X_train, y_train):
  """
  Passes possible combinations
  Saves models in a pickle file
  """
  # all combinatiosn
  all_combi = list(combinations(range(10),2))
  # save the models
  models = {}
  #for combination in combinations2:
  for i, (d1, d2) in enumerate(all_combi):
    weights, scaler = train_svm_model(d1, d2, X_train, y_train)
    key = f'{d1}_{d2}'
    models[key] = {
        'w': weights[0],
        'w0': weights[1],
        'scaler': scaler
    }
  # save the models
  classifier_package = {
      'models': models,
      'combinations': all_combi
  }
  # pickle the models
  with open("svm_classifier.pkl", "wb") as f:
      pickle.dump(classifier_package, f)

def svm_predict(unknown_digit,  model_file="svm_classifier.pkl"):
  """
  Loads trained models
  Predicts the digit
  Returns the predicted digit
  """
  # load the models
  with open(model_file, "rb") as f:
    model_datas = pickle.load(f)
  # unpack the model datas
  models_dict = model_datas['models']
  combis = model_datas['combinations']
  votes = {digit:0 for digit in range(10)}
  confidence_vals = {digit:0 for digit in range(10)}
  hydric_evalution = {}
  # reshape the sample
  flat_digit = unknown_digit.reshape(1,-1)
  # go throught the models
  for d1,d2 in combis:
    key_open = f'{d1}_{d2}'
    model_info = models_dict[key_open]
    w = model_info['w']
    w0 = model_info['w0']
    scaler = model_info['scaler']
    # flat_digit = unknown_digit.reshape(1, -1)
    # standardize
    standardized_digit = scaler.transform(flat_digit)
    score = np.dot(standardized_digit, w) + w0
    confidence = abs(score)
    if score < 0: # - 1
      votes[d1] += 1
      confidence_vals[d1] += confidence
    else:
      votes[d2] += 1
      confidence_vals[d2] += confidence
  # compose a hybrid score
  for digit in range(10):
    hydric_evalution[digit] = votes[digit] * (1+confidence_vals[digit]/100)
  # determine the best prediction
  predicted_digit = max(hydric_evalution, key=hydric_evalution.get)
  return predicted_digit