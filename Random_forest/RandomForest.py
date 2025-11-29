from DecisionTree import DecisionTree
import numpy as np
from collections import Counter

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_feature=None,bootstrap_size=None):
        self.n_trees = n_trees
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.n_features=n_feature
        self.trees = []
        self.bootstrap_size = bootstrap_size
    
    # The training function.
    def fit(self, X, y):
        # Again error prevention as in DT, but for the bootstrap_size parameter
        self.bootstrap_size = X.shape[0] if not self.bootstrap_size else min(X.shape[0],self.bootstrap_size)

        self.trees = []
        for i in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth,
                            min_samples_split=self.min_samples_split,
                            n_features=self.n_features)
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    # This is one part of the randomness of a random forest. A subset of the samples is used on each tree.
    # to introduce randomness in the classification, making it more general. 
    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        # replace = true allows for multiple entries of same
        idxs = np.random.choice(n_samples, self.bootstrap_size, replace=True) 
        return X[idxs], y[idxs]

    # Same function as was seen in decision tree, counts most commong label.
    # It's now used to determine the majority vote of ALL the trees in the forest for 
    # the final prediction, instead of just one nodes majority label.
    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    # Predict function retrieves all of the treest predictions for the test samples.
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # the predictions are retrieved as a lists of predictions for samples in a list.
        # This converts it so each sublist is instead every trees predictions on the same sample.
        tree_preds = np.swapaxes(predictions, 0, 1)
        
        # When the results are as lists of predictions per sample, choose the majority vote:
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])
        return predictions