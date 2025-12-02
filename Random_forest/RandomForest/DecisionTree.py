# credit source:
# https://github.com/AssemblyAI-Community/Machine-Learning-From-Scratch/tree/main/05%20Random%20Forests
import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    # When stopping criteria is met, the node is assigned a value -> leaf node, hence:    
    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split=min_samples_split # condition to stop when a node has less samples.
        self.max_depth=max_depth # condition to stop when max depth of a tree is reached.
        self.n_features=n_features # amount of features used to feature sampling in a random forest.
        self.root=None

    def fit(self, X, y):
        # Error prevention:
        # If there is no feature size given, use training data feature amount
        # and if there is a feature size given, choose it as long as it is within boundaries.
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1],self.n_features)
        self.root = self._grow_tree(X, y)

    # Recursively creates the tree structure.
    # After this function is triggered from the root, it loops on each branch 
    # until one of the stopping criteria is met.
    def _grow_tree(self, X, y, depth=0):
        # The amounts of samples and features
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # Check the stopping criteria
        if (depth>=self.max_depth or n_labels==1 or n_samples<self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Choosing randomly the features
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # Find the best split
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        # Create child nodes according to the best split
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feature, best_thresh, left, right)


    # This function calculates which split is the best, by comparing the informationg gain
    # of different feature values being used as the threshold
    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            # Testing each feature value as the threshold to see the best split.
            # So basically in this project, which moment of a hand writing stroke
            # separates the most samples.
            thresholds = np.unique(X_column)

            # calculate the information gain and loop through all of them to select the best.
            for thr in thresholds:
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold


    def _information_gain(self, y, X_column, threshold):
        # parent entropy
        parent_entropy = self._entropy(y)

        # create children
        # Splitting the values into children according to the threshold
        left_idxs, right_idxs = self._split(X_column, threshold)

        # If no division can be done(pure node), the IG is logically 0
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        # calculate the weighted avg. entropy of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        # calculate the IG
        # As can be seen, IG is means how much the entropy is decreased by the split.
        information_gain = parent_entropy - child_entropy
        return information_gain

    # This function separates the sample indexes based on their separator feature value vs threshold.
    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs


    # Formula for entropy:
    # E = -sum(p(X) * log2(p(X)))
    # So the percentages of of each different label in a node.
    # The entropy basically calculates how homogenous the node is.
    # Bincount counts the amount of each label appears, dividing it by len(y) converts to probability.
    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p>0])

    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    # for predicting results, this function "traverses" the tree down recursively.
    # by basically selecting nodes from the built tree, selecting based on how the given sample x
    # compares to each nodes threshold value until it reaches a leaf node.
    # from that leaf node it will deduce the output, either pure node(1 label), or majority label.
    # The afore mentioned label a leaf node has is produced on stop condition by _most_common_label().
    def _traverse_tree(self, x, node):
        # if the current node is leaf, return it, as it has the prediction.
        if node.is_leaf_node():
            return node.value

        # If the input x value is smaller than threshold, select left node, otherwise select right.
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
        