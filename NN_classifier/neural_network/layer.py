import numpy as np
from numpy.typing import NDArray
from .activation import Activation

# Define neural network layer parent class
class Layer:
    def __init__(self): pass

    # Define forward-propagation method signature
    def forward(self, X: NDArray) -> NDArray: pass

    # Define backward-propagation method signature
    def backward(self, grad: NDArray) -> NDArray: pass

    @property # Shortcut for checking layer input and output size
    def shape(self) -> tuple[int, int]: return self.weights.shape

    # Forward-method shortcut
    def __call__(self, X: NDArray) -> NDArray: return self.forward(X)

# Define fully connected layer
class Dense(Layer): 
    def __init__(self, weights: NDArray, biases: NDArray, activation_class: Activation): 
        assert weights.ndim == 2 and len(biases) == biases.size, "weights must be a matrix and biases a vector"
        assert weights.shape[1] == len(biases), "weights and biases must have matching dimensionality"
        super(Dense, self).__init__()
        self.weights, self.biases = weights, biases.flatten()
        self.grad_weights, self.grad_biases = np.zeros_like(self.weights), np.zeros_like(self.biases)
        self.activation = activation_class() # Instantialize activation class

    def forward(self, X: NDArray) -> NDArray: 
        assert X.ndim == 2 and X.shape[1] == self.weights.shape[0], "X has invalid dimensionality"
        self.X = X # Store inputs for backward pass
        return self.activation(X @ self.weights + self.biases)
    
    def backward(self, grad: NDArray) -> NDArray: 
        assert self.X is not None, 'forward pass needs to be made before backwards pass'   
        assert grad.ndim==2 and grad.shape==(len(self.X), len(self.biases)), 'grad has invalid dimensionality'        
        grad_activation = self.activation.backward(grad) # Compute the gradient of the activation wrt. loss
        self.grad_weights = self.X.T @ grad_activation # Compute the gradient of weights wrt. loss
        self.grad_biases = np.sum(grad_activation, axis=0) # Compute the gradient of biases wrt. loss
        return grad_activation @ self.weights.T # Return the gradient of inputs wrt. loss
    