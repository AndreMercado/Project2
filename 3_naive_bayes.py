"""
Naive Bayes classifier for binary/continuous features.

For each class c and feature index i we learn:
  P(feature_i = 1 | class = c)  (stored as log-likelihood)

Laplace smoothing (k=1) avoids zero probabilities.

For continuous grid features (floats), we binarize at threshold 0.5 before
building the likelihood tables — keeping the same interface as binary pixels.
"""

import math


class NaiveBayes:
    def __init__(self, k=1.0):
        self.k = k          # Laplace smoothing
        self.classes = []
        self.log_prior = {}
        self.log_like_on = {}   # log P(f_i=1 | c)
        self.log_like_off = {}  # log P(f_i=0 | c)

    def _binarize(self, features):
        return [1 if f >= 0.5 else 0 for f in features]

    def train(self, feature_matrix, labels):
        self.classes = list(set(labels))
        n_features = len(feature_matrix[0])
        n_total = len(labels)

        count_class = {c: 0 for c in self.classes}
        count_on = {c: [0.0] * n_features for c in self.classes}

        for feats, label in zip(feature_matrix, labels):
            count_class[label] += 1
            bfeats = self._binarize(feats)
            for i, v in enumerate(bfeats):
                count_on[label][i] += v

        self.log_prior = {}
        self.log_like_on = {}
        self.log_like_off = {}

        for c in self.classes:
            self.log_prior[c] = math.log(count_class[c] / n_total)
            n_c = count_class[c]
            self.log_like_on[c] = []
            self.log_like_off[c] = []
            for i in range(n_features):
                p_on = (count_on[c][i] + self.k) / (n_c + 2 * self.k)
                self.log_like_on[c].append(math.log(p_on))
                self.log_like_off[c].append(math.log(1.0 - p_on))

    def predict_one(self, features):
        bfeats = self._binarize(features)
        best_class = None
        best_score = float('-inf')
        for c in self.classes:
            score = self.log_prior[c]
            for i, v in enumerate(bfeats):
                score += self.log_like_on[c][i] if v else self.log_like_off[c][i]
            if score > best_score:
                best_score = score
                best_class = c
        return best_class

    def predict(self, feature_matrix):
        return [self.predict_one(f) for f in feature_matrix]

    def accuracy(self, feature_matrix, labels):
        preds = self.predict(feature_matrix)
        correct = sum(p == l for p, l in zip(preds, labels))
        return correct / len(labels)
