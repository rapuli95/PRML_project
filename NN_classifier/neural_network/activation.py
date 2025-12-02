import numpy as np
from numpy.typing import NDArray

# Define activation function parent class
class Activation: 
    def __init__(self): pass

    # Define forward propagation method signature
    def forward(self, X: NDArray) -> NDArray: pass

    # Define backwards propagation method signature
    def backward(self, grad: NDArray) -> NDArray: pass
    
    # Define forward-method shortcut 
    def __call__(self, X: NDArray) -> NDArray: return self.forward(X)

# Define ReLU activation
class ReLU(Activation): 
    def __init__(self): 
        super(ReLU, self).__init__()

    def forward(self, X: NDArray) -> NDArray:
        self.X = X
        return np.maximum(0, X)

    def backward(self, grad: NDArray) -> NDArray: 
        assert self.X is not None, 'forward pass needs to be made before backwards pass'
        return (self.X > 0) * grad
    
    def __repr__(self) -> str: 
        return "ReLU"

# Define Softmax activation    
class Softmax(Activation):
    def __init__(self): 
        super(Softmax, self).__init__()
    
    def forward(self, X: NDArray) -> NDArray:
        exp_X = np.exp(X - np.max(X, axis=-1, keepdims=True))
        self.probs = exp_X / np.sum(exp_X, axis=-1, keepdims=True)
        return self.probs

    def backward(self, grad: NDArray) -> NDArray:
        assert self.probs is not None, 'forward pass needs to be made before backwards pass'
        dot_product = np.sum(grad * self.probs, axis=-1, keepdims=True)
        return self.probs * (grad - dot_product)
    
    def __repr__(self) -> str: 
        return "Softmax"
