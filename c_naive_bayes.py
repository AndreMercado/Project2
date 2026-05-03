import math
from typing import Any

# The threshold used to decide if a feature value counts as "on" or "off".
# Feature values above 0.5 are treated as filled pixels.
BINARIZE_THRESHOLD: float = 0.5

# Laplace smoothing value added to every count to avoid zero probabilities.
# Without this, a feature never seen in training would give a probability of 0,
# which would make the entire prediction zero regardless of other features.
LAPLACE_SMOOTHING: float = 1.0


def binarize_features(features: list[float]) -> list[int]:
    # Step 36: Convert each feature value to either 0 or 1.
    # Naive Bayes needs binary features to learn "is this pixel on or off per class?"
    binary_values: list[int] = []
    for value in features:
        pixel_is_on: bool = value >= BINARIZE_THRESHOLD
        if pixel_is_on:
            binary_values.append(1)
        else:
            binary_values.append(0)
    return binary_values


def count_feature_occurrences(feature_matrix: list[list[float]], labels: list[Any], classes: list[Any], num_features: int) -> tuple[dict[Any, int], dict[Any, list[float]]]:
    # Step 35: Start with empty count tables, then tally counts from every training sample.
    # class_count tracks how many samples belong to each class.
    # feature_count tracks how many times each feature was ON for each class.
    class_count:   dict[Any, int]         = {label_class: 0                    for label_class in classes}
    feature_count: dict[Any, list[float]] = {label_class: [0.0] * num_features for label_class in classes}

    for features, label in zip(feature_matrix, labels):
        class_count[label] += 1
        binary_features: list[int] = binarize_features(features)
        for feature_index, pixel_is_on in enumerate(binary_features):
            feature_count[label][feature_index] += pixel_is_on

    return class_count, feature_count


def compute_log_likelihoods(class_count: dict[Any, int], feature_count: dict[Any, list[float]], classes: list[Any], num_features: int) -> tuple[dict[Any, list[float]], dict[Any, list[float]]]:
    # Step 39: For each class and each feature, compute the log probability
    # that the feature is ON given that class: log P(feature_i = 1 | class).
    # We store log probabilities instead of raw probabilities to avoid
    # underflow when multiplying many small numbers together.
    log_likelihood_on:  dict[Any, list[float]] = {}
    log_likelihood_off: dict[Any, list[float]] = {}

    for label_class in classes:
        samples_in_class: int = class_count[label_class]
        log_likelihood_on[label_class]  = []
        log_likelihood_off[label_class] = []

        for feature_index in range(num_features):
            # Step 40: Apply Laplace smoothing and compute the probability.
            # Laplace smoothing prevents zero probabilities for unseen feature values.
            probability_feature_on: float = (
                (feature_count[label_class][feature_index] + LAPLACE_SMOOTHING)
                / (samples_in_class + 2 * LAPLACE_SMOOTHING)
            )
            log_likelihood_on[label_class].append(math.log(probability_feature_on))
            log_likelihood_off[label_class].append(math.log(1.0 - probability_feature_on))

    return log_likelihood_on, log_likelihood_off


def naive_bayes_train(feature_matrix: list[list[float]], labels: list[Any]) -> dict[str, Any]:
    # Step 33: Train the Naive Bayes model by building all probability tables.
    classes:       list[Any] = list(set(labels))
    num_features:  int       = len(feature_matrix[0])
    total_samples: int       = len(labels)

    # Step 34: Count how often each feature is ON for each class.
    class_count, feature_count = count_feature_occurrences(
        feature_matrix, labels, classes, num_features
    )

    # Step 37: Compute the log prior probability for each class.
    # This is simply log(how often this class appears in training data).
    log_prior: dict[Any, float] = {
        label_class: math.log(class_count[label_class] / total_samples)
        for label_class in classes
    }

    # Step 38: Compute the log likelihood tables for every feature and class.
    log_likelihood_on, log_likelihood_off = compute_log_likelihoods(
        class_count, feature_count, classes, num_features
    )

    # Step 41: Bundle everything into a model dictionary and return it.
    model: dict[str, Any] = {
        'classes':            classes,
        'log_prior':          log_prior,
        'log_likelihood_on':  log_likelihood_on,
        'log_likelihood_off': log_likelihood_off,
    }
    return model


def score_class(model: dict[str, Any], binary_features: list[int], label_class: Any) -> float:
    # Step 45: Compute the total log probability score for one class.
    # Start with the log prior, then add the log likelihood of each feature.
    # Adding logs is the same as multiplying probabilities, but numerically safer.
    score: float = model['log_prior'][label_class]
    for feature_index, pixel_is_on in enumerate(binary_features):
        if pixel_is_on:
            score += model['log_likelihood_on'][label_class][feature_index]
        else:
            score += model['log_likelihood_off'][label_class][feature_index]
    return score


def naive_bayes_predict_one(model: dict[str, Any], features: list[float]) -> Any:
    # Step 44: Predict the class for a single image.
    # Score every class and return the one with the highest score.
    binary_features: list[int] = binarize_features(features)
    best_class: Any   = None
    best_score: float = float('-inf')

    for label_class in model['classes']:
        score: float    = score_class(model, binary_features, label_class)
        is_new_best: bool = score > best_score
        if is_new_best:
            best_score = score
            best_class = label_class

    return best_class


def naive_bayes_predict(model: dict[str, Any], feature_matrix: list[list[float]]) -> list[Any]:
    # Step 43: Predict the class for every image in the dataset.
    predictions: list[Any] = []
    for features in feature_matrix:
        predictions.append(naive_bayes_predict_one(model, features))
    return predictions
