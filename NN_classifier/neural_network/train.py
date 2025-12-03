import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from .neural_network import NeuralNetwork
from .loss import Loss, CategoricalCrossEntropySoftmax

# Define neural network training loop 
def train(model: NeuralNetwork, 
          X_train: NDArray, y_train: NDArray, X_val: NDArray|None=None, y_val: NDArray|None=None,
          criterion: Loss=CategoricalCrossEntropySoftmax, 
          learning_rate: float=0.001, decay: float=0.001, batch_size: int=10, 
          max_iter: int=1000, tol: float=0.001, min_loss: float=1e-10, verbose: bool=False
    ) -> NDArray: 
    
    # Check if separate validation sets were provided 
    if X_val is None or y_val is None: 
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)
    
    # Initialize arrays for tracking model loss and accuracy
    y_pred = model.forward(X_val)
    loss_history = [criterion.forward(y_val, y_pred)]
    accuracy_history = [accuracy_score(np.argmax(y_val, axis=1), np.argmax(y_pred, axis=1))]

    # Iterate over epochs
    for _ in (tqdm(range(max_iter)) if verbose else range(max_iter)): 

        # Compute shuffled training data indices
        shuffled_idx = np.random.permutation(len(X_train))

        # Iterate over batches
        for batch_idx in [shuffled_idx[i:i+batch_size] for i in range(0, len(X_train), batch_size)]: 

            # Forward propagate a batch
            X_batch, y_batch = X_train[batch_idx], y_train[batch_idx]
            y_pred = model.forward(X_batch)
            
            # Backwards propagate gradients
            grad = criterion.backward(y_batch, y_pred)
            model.backward(grad)

            # Update model parameters
            for layer in model.layers: 
                layer.weights -= learning_rate * (layer.grad_weights + decay * layer.weights)
                layer.biases -= learning_rate * layer.grad_biases # + decay * layer.biases)
               
        # Record model loss and accuracy
        y_pred = model.forward(X_val)
        loss_history.append(criterion.forward(y_val, y_pred))
        accuracy_history.append(accuracy_score(np.argmax(y_val, axis=1), np.argmax(y_pred, axis=1)))

        # Convergence check 
        loss_diff = abs(loss_history[-2] - loss_history[-1])
        if loss_diff <= tol or loss_history[-1] <= min_loss: break
    
    # Return criterion history and accuracy history
    return np.array(loss_history), np.array(accuracy_history)