import pandas as pd
import numpy as np

class DecisionTreeID3:
    def __init__(self):
        self.tree = {}

    def entropy(self, y):
        values, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def information_gain(self, X, y, feature):
        total_entropy = self.entropy(y)
        values, counts = np.unique(X[:, feature], return_counts=True)
        weighted_entropy = np.sum((counts[i] / np.sum(counts)) * self.entropy(y[X[:, feature] == values[i]]) for i in range(len(values)))
        return total_entropy - weighted_entropy

    def best_feature(self, X, y):
        feature_gains = [self.information_gain(X, y, feature) for feature in range(X.shape[1])]
        return np.argmax(feature_gains)

    def build_tree(self, X, y, features):
        if len(np.unique(y)) == 1:
            return np.unique(y)[0]
        elif len(features) == 0:
            return np.unique(y)[np.argmax(np.unique(y, return_counts=True)[1])]

        best_feature = self.best_feature(X, y)
        tree = {features[best_feature]: {}}

        feature_values = np.unique(X[:, best_feature])
        for value in feature_values:
            sub_X = X[X[:, best_feature] == value]
            sub_y = y[X[:, best_feature] == value]
            sub_features = np.delete(features, best_feature)

            subtree = self.build_tree(sub_X, sub_y, sub_features)
            tree[features[best_feature]][value] = subtree

        return tree

    def fit(self, X, y):
        features = np.arange(X.shape[1])
        self.tree = self.build_tree(X, y, features)

    def predict_sample(self, sample, tree):
        if not isinstance(tree, dict):
            return tree

        feature = list(tree.keys())[0]
        subtree = tree[feature][sample[feature]]
        return self.predict_sample(sample, subtree)

    def predict(self, X):
        return np.array([self.predict_sample(sample, self.tree) for sample in X])
