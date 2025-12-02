# PRML_project

Three methods to approach the digit classification problem

## MAIN IMPLEMENTATION FILES

- `digits_classify.ipynb`: the main function implementation in notebook format

- `digits_classify.py`: same implementation as in `digits_classify.ipynb`, but as a runnable script. Function `digits_classify` can also be imported to another file from here.

- `requirements.txt`: project requirements

## Other files

- `NN_classifier.ipynb`: neural network classifier initialization, training, testing and performance evaluation

- `RF_training.ipynb`: random forest model initialization and training

- `RF_for_testing.ipynb`: random forest testing and performance evaluation

- `run_svm.py`: SVM model initialization, training, testing and performance evaluation

## Subdirectories

- `training_data`: contains the training data files

- `trained_models`: contains trained model files, which are utilized by function `classify_digits`

- `NN_classifier`: neural network classifier implementation code and some results

- `Random_forest`: random forest implementation code

- `SVM`: SVM model implementation code

## 1) NEURAL NETWORKS implementation
The folder "neural_network" includes seven files:

**__init__.py**
- contains the imports from the other files

**activation.py**
- class Activation
- class RelU
- class Softmax
-> each contains forward & backward propagation function  
⟶ import numpy, numpy.typing

**layer.py**
- class Layer has forward & backward propagation
- class Dense (fully connected layer) has forward and backward propagation function  
⟶ import from .activation

**loss.py**
- class Loss has forward- and back propagation and evaluate function
- class CategoralCrossEntropySoftMax has forward-and backpropagation function  
⟶ import numpy

**neural_network.py**
- class NeuralNetwork has forward- and backpropagation function
- class NeuralNetworkClassifier has a prediction function  
⟶ imports from numpy and .layer

**train.py**
- function train that iterates over epochs and batchs. Forward propagates a batch and backward propagates
gradients. Returns criterion history and accuracy history.  
⟶ import from .neural_network and .loss & numpy, tqdm, sklearn

**utils.py**
- function one_hot_encode, returns labels encoded  
⟶  import from numpy

The folder "main" includes two files:

**nn_classifier.ipynb**
- loads the data
- applies PCA (3D to 2D)
- converts data to images
- flattens images to vectors and labels to an array
- splits data to training and testing sets
- trains models
- combines the prepreprocessing pipeline and NN classifier to a single function  
⟶ import os, np, pd, matplotlib, sklearn, typing, neural_network.py

**preprocessing_pipeline.ipynb**
- loads the data
- saves every samples as .png
- applies PCA (3D to 2D)
- converts data to images
- saves the images

## 2) Support Vector Machine
The folder "SVM" includes one folder and one file:
**svm_classifier.ipynb**
- interpolates the data
- implements SVM, creating binary classification for all combinations
0-9
- takes the new digit and compares it trained models
- predicts the label of the digit
⟶ import pandas, numpy, scikit-learn, cvxopt, itertools, random
**parquets**
- each parquet contains the 100 samples of the digit


## 3) Random Forest

