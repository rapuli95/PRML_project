import numpy as np
from numpy import ndarray

class Loss:
    def forward(self, y_true: ndarray, y_pred: ndarray) -> ndarray: ...
    
    def backward(self, y_true: ndarray, y_pred: ndarray) -> ndarray: ...
    
    def evaluate(self, y_true: ndarray, y_pred: ndarray) -> tuple[float, ndarray]: 
        return self.forward(y_true, y_pred), self.backward(y_true, y_pred)
    
    def __call__(self, y_true: ndarray, y_pred: ndarray) -> ndarray: 
        return self.forward(y_true, y_pred)
    
class CategoricalCrossEntropySoftmax(Loss):
    def forward(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        losses = -np.sum(y_true * np.log(y_pred), axis=1)
        return np.mean(losses)

    def backward(self, y_true, y_pred):
        batch_size = y_true.shape[0]
        return (y_pred - y_true) / batch_size