"""
Multi-class Perceptron classifier.

Weight update rule (when prediction != true label y):
  w[y]  += feature_vector
  w[y'] -= feature_vector   (y' = predicted class)

Training runs for a fixed number of epochs with optional random shuffling.
"""

import random


class Perceptron:
    def __init__(self, epochs=10):
        self.epochs = epochs
        self.classes = []
        self.weights = {}   # class -> list of floats
        self.bias = {}      # class -> float

    def _score(self, features, c):
        w = self.weights[c]
        return sum(w[i] * features[i] for i in range(len(features))) + self.bias[c]

    def _predict_one_raw(self, features):
        best_class = None
        best_score = float('-inf')
        for c in self.classes:
            s = self._score(features, c)
            if s > best_score:
                best_score = s
                best_class = c
        return best_class

    def train(self, feature_matrix, labels):
        self.classes = list(set(labels))
        n_features = len(feature_matrix[0])

        self.weights = {c: [0.0] * n_features for c in self.classes}
        self.bias = {c: 0.0 for c in self.classes}

        indices = list(range(len(feature_matrix)))
        for _ in range(self.epochs):
            random.shuffle(indices)
            for idx in indices:
                features = feature_matrix[idx]
                true_label = labels[idx]
                pred_label = self._predict_one_raw(features)
                if pred_label != true_label:
                    wt = self.weights[true_label]
                    wp = self.weights[pred_label]
                    for i in range(n_features):
                        wt[i] += features[i]
                        wp[i] -= features[i]
                    self.bias[true_label] += 1.0
                    self.bias[pred_label] -= 1.0

    def predict_one(self, features):
        return self._predict_one_raw(features)

    def predict(self, feature_matrix):
        return [self.predict_one(f) for f in feature_matrix]

    def accuracy(self, feature_matrix, labels):
        preds = self.predict(feature_matrix)
        correct = sum(p == l for p, l in zip(preds, labels))
        return correct / len(labels)
