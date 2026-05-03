import random
from typing import Any

# Number of full passes through the training data.
# More epochs means more chances to correct mistakes, but takes longer.
TRAINING_EPOCHS: int = 10


def compute_class_score(weights: dict[Any, list[float]], bias: dict[Any, float], features: list[float], label_class: Any) -> float:
    # Step 54: Compute the score for one class given one image's features.
    # Score = bias + sum of (weight * feature value) for every feature.
    # The class with the highest score wins the prediction.
    score: float = bias[label_class]
    for feature_index, feature_value in enumerate(features):
        score += weights[label_class][feature_index] * feature_value
    return score


def predict_label(weights: dict[Any, list[float]], bias: dict[Any, float], classes: list[Any], features: list[float]) -> Any:
    # Step 53: Score every class and return the one with the highest score.
    # This is the perceptron's prediction for a single image.
    best_class: Any = None
    best_score: float = float('-inf')

    for label_class in classes:
        score: float      = compute_class_score(weights, bias, features, label_class)
        is_new_best: bool = score > best_score
        if is_new_best:
            best_score = score
            best_class = label_class

    return best_class


def update_weights(weights: dict[Any, list[float]], bias: dict[Any, float], features: list[float], true_label: Any, predicted_label: Any) -> None:
    # Step 56: Correct the weights when the perceptron made a wrong prediction.
    # Increase the weights for the true class so it scores higher next time.
    # Decrease the weights for the wrong class so it scores lower next time.
    for feature_index, feature_value in enumerate(features):
        weights[true_label][feature_index]      += feature_value
        weights[predicted_label][feature_index] -= feature_value
    bias[true_label]      += 1.0
    bias[predicted_label] -= 1.0


def perceptron_train(feature_matrix: list[list[float]], labels: list[Any]) -> dict[str, Any]:
    # Step 50: Set up one weight vector per class, all starting at zero.
    # Each weight vector has one weight per feature.
    classes: list[Any]        = list(set(labels))
    num_features: int         = len(feature_matrix[0])

    weights: dict[Any, list[float]] = {label_class: [0.0] * num_features for label_class in classes}
    bias: dict[Any, float]          = {label_class: 0.0                  for label_class in classes}

    sample_indices: list[int] = list(range(len(feature_matrix)))

    # Step 51: Repeat training for multiple epochs.
    # Each epoch shuffles the data to prevent order-dependent bias.
    for _ in range(TRAINING_EPOCHS):
        random.shuffle(sample_indices)

        # Step 52: For each training sample, predict and update if wrong.
        for sample_index in sample_indices:
            features: list[float]  = feature_matrix[sample_index]
            true_label: Any        = labels[sample_index]
            predicted_label: Any   = predict_label(weights, bias, classes, features)

            # Step 55: Only update weights when the prediction is wrong.
            # Correct predictions leave the weights unchanged.
            prediction_is_wrong: bool = predicted_label != true_label
            if prediction_is_wrong:
                update_weights(weights, bias, features, true_label, predicted_label)

    # Step 57: Bundle the learned weights and biases into a model and return it.
    model: dict[str, Any] = {
        'classes': classes,
        'weights': weights,
        'bias':    bias,
    }
    return model


def perceptron_predict(model: dict[str, Any], feature_matrix: list[list[float]]) -> list[Any]:
    # Step 58: Predict the class for every image using the learned weights.
    predictions: list[Any] = []
    for features in feature_matrix:
        predicted: Any = predict_label(
            model['weights'], model['bias'], model['classes'], features
        )
        predictions.append(predicted)
    return predictions
