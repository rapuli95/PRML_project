import numpy as np
from numpy.typing import NDArray
from .layer import Layer

# Define neural network parent class
class NeuralNetwork: 
    def __init__(self, *layers: Layer): 
        self.layers = layers

    # Define the forward-propagation method
    def forward(self, X: NDArray) -> NDArray: 
        for layer in self.layers: 
            X = layer.forward(X)
        return X
    
    # Define the backward-propagation method
    def backward(self, grad: NDArray) -> NDArray:
        for layer in reversed(self.layers): 
            grad = layer.backward(grad)

    @property # Define shortcut for neural network input and output size
    def shape(self) -> tuple[int, int]: 
        return self.layers[0].shape[0], self.layers[-1].shape[1]

# Define neural network classifier
class NeuralNetworkClassifier(NeuralNetwork): 
    def __init__(self, *layers: Layer): 
        super(NeuralNetworkClassifier, self).__init__(*layers)

    # The predict method predicts the class with the highest probability   
    def predict(self, X: NDArray) -> NDArray: 
        return np.argmax(self.forward(X), axis=1)