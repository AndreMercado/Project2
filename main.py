"""
Main script for CS4346 Project 2 - Image Classification.

Runs Naive Bayes and Perceptron on digit and face datasets at training
percentages 10%-100% (step 10), with 5 random trials each.
Reports mean accuracy, std deviation, and mean training runtime.

Usage:
    python main.py
"""

import random
import time
import math

from data_loader import load_dataset
from features import extract_features
from naive_bayes import NaiveBayes
from perceptron import Perceptron

DIGIT_DATA_DIR = r'digitdata'
FACE_DATA_DIR  = r'facedata'

PERCENTAGES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
N_TRIALS = 5


def mean(vals):
    return sum(vals) / len(vals)


def std(vals):
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


def build_features(images):
    return [extract_features(img) for img in images]


def evaluate(classifier_cls, train_feats, train_labels,
             test_feats, test_labels, pct, n_trials, **kwargs):
    n_train = len(train_feats)
    k = max(1, int(n_train * pct / 100))

    accs = []
    times = []
    for _ in range(n_trials):
        indices = random.sample(range(n_train), k)
        subset_feats  = [train_feats[i]  for i in indices]
        subset_labels = [train_labels[i] for i in indices]

        clf = classifier_cls(**kwargs)
        t0 = time.time()
        clf.train(subset_feats, subset_labels)
        elapsed = time.time() - t0

        acc = clf.accuracy(test_feats, test_labels)
        accs.append(acc)
        times.append(elapsed)

    return mean(accs), std(accs), mean(times)


def run_experiment(dataset_name, data_dir, split_train='train', split_test='test'):
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name.upper()}")
    print(f"{'='*60}")

    print("Loading data...", flush=True)
    train_images, train_labels = load_dataset(data_dir, dataset_name, 'train')
    test_images,  test_labels  = load_dataset(data_dir, dataset_name, 'test')

    print(f"  Train: {len(train_images)} images | Test: {len(test_images)} images")
    print("Extracting features...", flush=True)
    train_feats = build_features(train_images)
    test_feats  = build_features(test_images)

    print(f"  Feature vector length: {len(train_feats[0])}")

    header = f"{'%':>4}  {'NB Acc':>8}  {'NB Std':>8}  {'NB Time':>8}  " \
             f"{'Pct Acc':>8}  {'Pct Std':>8}  {'Pct Time':>8}"
    print(f"\n{header}")
    print('-' * len(header))

    for pct in PERCENTAGES:
        nb_acc, nb_std, nb_time = evaluate(
            NaiveBayes, train_feats, train_labels,
            test_feats, test_labels, pct, N_TRIALS
        )
        pt_acc, pt_std, pt_time = evaluate(
            Perceptron, train_feats, train_labels,
            test_feats, test_labels, pct, N_TRIALS, epochs=10
        )
        print(f"{pct:>3}%  "
              f"{nb_acc*100:>7.2f}%  {nb_std*100:>7.2f}%  {nb_time:>7.3f}s  "
              f"{pt_acc*100:>7.2f}%  {pt_std*100:>7.2f}%  {pt_time:>7.3f}s")


def main():
    random.seed(42)
    run_experiment('digits', DIGIT_DATA_DIR)
    run_experiment('faces',  FACE_DATA_DIR)


if __name__ == '__main__':
    main()
