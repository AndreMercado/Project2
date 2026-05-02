import math

# The threshold used to decide if a feature value counts as "on" or "off".
# Feature values above 0.5 are treated as filled pixels.
BINARIZE_THRESHOLD = 0.5

# Laplace smoothing value added to every count to avoid zero probabilities.
# Without this, a feature never seen in training would give a probability of 0,
# which would make the entire prediction zero regardless of other features.
LAPLACE_SMOOTHING  = 1.0


def binarize_features(features):
    # Step 1: Convert each feature value to either 0 or 1.
    # Naive Bayes needs binary features to learn "is this pixel on or off per class?"
    binary_values = []
    for value in features:
        pixel_is_on = value >= BINARIZE_THRESHOLD
        if pixel_is_on:
            binary_values.append(1)
        else:
            binary_values.append(0)
    return binary_values


def count_feature_occurrences(feature_matrix, labels, classes, num_features):
    # Step 2: Start with empty count tables, then tally counts from every training sample.
    # class_count tracks how many samples belong to each class.
    # feature_count tracks how many times each feature was ON for each class.
    class_count   = {label_class: 0                    for label_class in classes}
    feature_count = {label_class: [0.0] * num_features for label_class in classes}

    for features, label in zip(feature_matrix, labels):
        class_count[label] += 1
        binary_features = binarize_features(features)
        for feature_index, pixel_is_on in enumerate(binary_features):
            feature_count[label][feature_index] += pixel_is_on

    return class_count, feature_count


def compute_log_likelihoods(class_count, feature_count, classes, num_features):
    # Step 4: For each class and each feature, compute the log probability
    # that the feature is ON given that class: log P(feature_i = 1 | class).
    # We store log probabilities instead of raw probabilities to avoid
    # underflow when multiplying many small numbers together.
    log_likelihood_on  = {}
    log_likelihood_off = {}

    for label_class in classes:
        samples_in_class = class_count[label_class]
        log_likelihood_on[label_class]  = []
        log_likelihood_off[label_class] = []

        for feature_index in range(num_features):
            # Step 5: Apply Laplace smoothing and compute the probability.
            # Laplace smoothing prevents zero probabilities for unseen feature values.
            probability_feature_on = (
                (feature_count[label_class][feature_index] + LAPLACE_SMOOTHING)
                / (samples_in_class + 2 * LAPLACE_SMOOTHING)
            )
            log_likelihood_on[label_class].append(math.log(probability_feature_on))
            log_likelihood_off[label_class].append(math.log(1.0 - probability_feature_on))

    return log_likelihood_on, log_likelihood_off


def naive_bayes_train(feature_matrix, labels):
    # Step 6: Train the Naive Bayes model by building all probability tables.
    classes       = list(set(labels))
    num_features  = len(feature_matrix[0])
    total_samples = len(labels)

    # Step 7: Count how often each feature is ON for each class.
    class_count, feature_count = count_feature_occurrences(
        feature_matrix, labels, classes, num_features
    )

    # Step 8: Compute the log prior probability for each class.
    # This is simply log(how often this class appears in training data).
    log_prior = {
        label_class: math.log(class_count[label_class] / total_samples)
        for label_class in classes
    }

    # Step 9: Compute the log likelihood tables for every feature and class.
    log_likelihood_on, log_likelihood_off = compute_log_likelihoods(
        class_count, feature_count, classes, num_features
    )

    # Step 10: Bundle everything into a model dictionary and return it.
    model = {
        'classes':            classes,
        'log_prior':          log_prior,
        'log_likelihood_on':  log_likelihood_on,
        'log_likelihood_off': log_likelihood_off,
    }
    return model


def score_class(model, binary_features, label_class):
    # Step 11: Compute the total log probability score for one class.
    # Start with the log prior, then add the log likelihood of each feature.
    # Adding logs is the same as multiplying probabilities, but numerically safer.
    score = model['log_prior'][label_class]
    for feature_index, pixel_is_on in enumerate(binary_features):
        if pixel_is_on:
            score += model['log_likelihood_on'][label_class][feature_index]
        else:
            score += model['log_likelihood_off'][label_class][feature_index]
    return score


def naive_bayes_predict_one(model, features):
    # Step 12: Predict the class for a single image.
    # Score every class and return the one with the highest score.
    binary_features = binarize_features(features)
    best_class = None
    best_score = float('-inf')

    for label_class in model['classes']:
        score = score_class(model, binary_features, label_class)
        if score > best_score:
            best_score = score
            best_class = label_class

    return best_class


def naive_bayes_predict(model, feature_matrix):
    # Step 13: Predict the class for every image in the dataset.
    predictions = []
    for features in feature_matrix:
        predictions.append(naive_bayes_predict_one(model, features))
    return predictions


