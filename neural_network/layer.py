import numpy as np
from numpy.typing import NDArray

# Define neural network layer parent class
class Layer:
    def __init__(self): pass

    # Define forward-propagation method
    def forward(self, X: NDArray) -> NDArray: pass

    # Define backward-propagation method
    def backward(self, grad: NDArray) -> NDArray: pass

    @property # Shortcut for checking layer input and output size
    def shape(self) -> tuple[int, int]: return self.weights.shape
    
    @property # Parameters shortcut for updating parameters in training loop 
    def parameters(self) -> list[NDArray]: pass

    @parameters.setter # Parameters setter for updating parameters in training loop 
    def parameters(self, new_params: list[NDArray]) -> list[NDArray]: pass
    
    @property # Gradients shortcut for training loop 
    def gradients(self) -> list[NDArray]: pass 

    # Forward-method shortcut
    def __call__(self, X: NDArray) -> NDArray: return self.forward(X)

# Define linear layer
class Linear(Layer): 
    def __init__(self, weights: NDArray, biases: NDArray): 
        assert weights.ndim == 2 and len(biases) == biases.size, "weights must be a matrix and biases a vector"
        assert weights.shape[1] == len(biases), "weights and biases must have matching dimensionality"
        super(Linear, self).__init__()
        self.weights, self.biases = weights, biases.flatten()
        self.grad_weights, self.grad_biases = np.zeros_like(self.weights), np.zeros_like(self.biases)

    def forward(self, X: NDArray) -> NDArray: 
        assert X.ndim == 2 and X.shape[1] == self.weights.shape[0], "X has invalid dimensionality"
        self.X = X # Store inputs for backward pass
        return X @ self.weights + self.biases
    
    def backward(self, grad: NDArray) -> NDArray: 
        assert self.X is not None, 'forward pass needs to be made before backwards pass'   
        assert grad.shape[-1] == len(self.biases), 'grad has invalid dimensionality'
        self.grad_weights = self.X.T @ grad #/ len(self.X) # Compute gradient of weights wrt. loss
        self.grad_biases = np.sum(grad, axis=0) # Compute gradient of biases wrt. loss
        return grad @ self.weights.T # Compute gradient of input wrt. loss
    
    @property 
    def parameters(self) -> list[NDArray]: 
        return self.weights, self.biases

    @parameters.setter
    def parameters(self, new_params: list[NDArray]) -> list[NDArray]: 
        self.weights, self.biases = new_params

    @property
    def gradients(self) -> list[NDArray]: 
        return self.grad_weights, self.grad_biases