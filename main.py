import random
import time
import statistics
from typing import Any, Callable

from a_data_loader import load_dataset
from b_features    import extract_features
from c_naive_bayes import naive_bayes_train, naive_bayes_predict
from d_perceptron  import perceptron_train, perceptron_predict

DIGIT_DATA_DIR:     str       = 'digitdata'
FACE_DATA_DIR:      str       = 'facedata'
TRAINING_PERCENTS:  list[int] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
TRIALS_PER_PERCENT: int       = 5


def compute_accuracy(predictions: list[Any], labels: list[Any]) -> float:
    # Step 46: Count how many predictions match the true labels and return the fraction.
    return sum(p == t for p, t in zip(predictions, labels)) / len(labels)


def sample_training_data(train_feature_vectors: list[list[float]], train_labels: list[Any], percent: int) -> tuple[list[list[float]], list[Any]]:
    # Step 31: Randomly pick a percentage of the training data.
    # We do this to test how well each classifier learns with less data.
    total_samples:    int       = len(train_feature_vectors)
    sample_size:      int       = max(1, int(total_samples * percent / 100))
    selected_indices: list[int] = random.sample(range(total_samples), sample_size)
    sampled_feature_vectors: list[list[float]] = [train_feature_vectors[index] for index in selected_indices]
    sampled_labels:          list[Any]         = [train_labels[index]          for index in selected_indices]
    return sampled_feature_vectors, sampled_labels


def evaluate_classifier(training_function: Callable, predict_function: Callable, train_data: tuple, test_data: tuple, percent: int) -> tuple[float, float, float]:
    # Step 29: Run multiple trials for one classifier at one training percentage.
    # Each trial randomly picks a different subset of training data.
    # We collect accuracy and training time from each trial.
    train_feature_vectors, train_labels = train_data
    test_feature_vectors,  test_labels  = test_data

    accuracies:     list[float] = []
    training_times: list[float] = []

    for _ in range(TRIALS_PER_PERCENT):
        # Step 30: Sample a random subset of the training data for this trial.
        sampled_feature_vectors, sampled_labels = sample_training_data(
            train_feature_vectors, train_labels, percent
        )

        # Step 32: Train the classifier and measure how long it takes.
        start_time:        float          = time.time()
        model:             dict[str, Any] = training_function(sampled_feature_vectors, sampled_labels)
        training_duration: float          = time.time() - start_time

        # Step 42: Predict on test data and compute accuracy.
        predictions: list[Any] = predict_function(model, test_feature_vectors)
        accuracy:    float     = compute_accuracy(predictions, test_labels)
        accuracies.append(accuracy)
        training_times.append(training_duration)

    # Step 47: Return mean accuracy, standard deviation, and mean training time.
    return statistics.mean(accuracies), statistics.stdev(accuracies), statistics.mean(training_times)


def print_result_row(percent: int, naive_bayes_result: tuple[float, float, float], perceptron_result: tuple[float, float, float]) -> None:
    # Step 59: Print one row of results for both classifiers at a given percentage.
    naive_bayes_accuracy, naive_bayes_standard_deviation, naive_bayes_time = naive_bayes_result
    perceptron_accuracy,  perceptron_standard_deviation,  perceptron_time  = perceptron_result
    print(
        f"{percent:>3}%  "
        f"{naive_bayes_accuracy * 100:>7.2f}%  {naive_bayes_standard_deviation * 100:>7.2f}%  {naive_bayes_time:>7.3f}s  "
        f"{perceptron_accuracy * 100:>7.2f}%  {perceptron_standard_deviation * 100:>7.2f}%  {perceptron_time:>7.3f}s"
    )


def print_table_header() -> None:
    # Step 27: Print the column labels for the results table.
    header: str = (
        f"{'%':>4}  {'NB Accuracy':>12}  {'NB Std Dev':>12}  {'NB Time':>8}  "
        f"{'Perc Accuracy':>14}  {'Perc Std Dev':>13}  {'Perc Time':>10}"
    )
    print(f"\n{header}")
    print('-' * len(header))


def load_and_extract_features(data_dir: str, dataset_name: str, split: str) -> tuple[list[list[float]], list[Any]]:
    # Step 4: Load raw images from disk, then convert each image into a feature vector.
    images, labels           = load_dataset(data_dir, dataset_name, split)
    feature_vectors: list[list[float]] = [extract_features(image) for image in images]
    return feature_vectors, labels


def run_experiment(dataset_name: str, data_dir: str) -> None:
    # Step 2: Run the full experiment for one dataset.
    # Trains and evaluates both classifiers at every training percentage.
    print(f"\n{'=' * 60}")
    print(f"  Dataset: {dataset_name.upper()}")
    print(f"{'=' * 60}")

    # Step 3: Load and featurize the training and test splits.
    print("Loading data and extracting features...")
    train_data = load_and_extract_features(data_dir, dataset_name, 'train')
    test_data  = load_and_extract_features(data_dir, dataset_name, 'test')

    train_features, _ = train_data
    print(f"  Train: {len(train_features)} images | Test: {len(test_data[0])} images")
    print(f"  Feature vector length: {len(train_features[0])}")

    print_table_header()

    # Step 28: For each training percentage, run trials for both classifiers
    # and print the results side by side.
    for percent in TRAINING_PERCENTS:
        naive_bayes_result = evaluate_classifier(
            naive_bayes_train, naive_bayes_predict, train_data, test_data, percent
        )
        perceptron_result = evaluate_classifier(
            perceptron_train, perceptron_predict, train_data, test_data, percent
        )
        print_result_row(percent, naive_bayes_result, perceptron_result)


def main() -> None:
    # Step 1: Run the experiment on both datasets.
    run_experiment('digits', DIGIT_DATA_DIR)
    run_experiment('faces',  FACE_DATA_DIR)


if __name__ == '__main__':
    main()
