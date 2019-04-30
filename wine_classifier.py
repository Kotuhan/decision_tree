#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skeleton code for CW2 submission.
We suggest you implement your code in the provided functions
Make sure to use our print_features() and print_predictions() functions
to print your results
"""

from __future__ import print_function

import argparse
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from utilities import load_data, print_features, print_predictions
from decision_tree import build_tree, classify

# you may use these colours to produce the scatter plots
CLASS_1_C = r'#3366ff'
CLASS_2_C = r'#cc3300'
CLASS_3_C = r'#ffc34d'

MODES = ['feature_sel', 'knn', 'alt', 'knn_3d', 'knn_pca']

################################################################################
################################################################################
# Plot 13x13 scatter plot containing all features and output selected 2 features


def feature_selection(train_set, train_labels, **kwargs):

    n_features = train_set.shape[1]
    fig, ax = plt.subplots(n_features, n_features)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99,
                        bottom=0.01, wspace=0.2, hspace=0.4)

    class_1_colour = r'#3366ff'
    class_2_colour = r'#cc3300'
    class_3_colour = r'#ffc34d'
    class_colours = np.array([class_1_colour, class_2_colour, class_3_colour])

    for x in range(0, n_features):
        for y in range(0, n_features):
            title = 'Features {} vs {}'.format(x+1, y+1)
            ax[x, y].scatter(train_set[:, x], train_set[:, y],
                             c=class_colours[train_labels-1])
            ax[x, y].set_title(title)

    # plt.show()
    selected_two_features = [9, 12]

    return selected_two_features
################################################################################
################################################################################
# Produce new data that contains only features [9,12]


def reduce_data(train_set, test_set, selected_features):
    train_set_red = train_set[:, selected_features]
    test_set_red = test_set[:, selected_features]
    return train_set_red, test_set_red


def calculate_accuracy(gt_labels, pred_labels):
    correct = 0.00
    for i in range(len(gt_labels)):
        if gt_labels[i] == pred_labels[i]:
            correct = correct + 1
    accuracy = correct/len(gt_labels)
    return accuracy


def calculate_confusion_matrix(gt_labels, pred_labels):
    size = len(np.unique(gt_labels))
    CM = np.zeros((size, size))
    for i in range(len(gt_labels)):
        CM[gt_labels[i]-1][pred_labels[i]-1] += 1
    for i in range(size):
        CM[i] /= np.sum(CM, axis=1)[i]
    print(CM)
    return CM


def plot_matrix(matrix, ax=None):
    """
    Displays a given matrix as an image.

    Args:
        - matrix: the matrix to be displayed
        - ax: the matplotlib axis where to overlay the plot.
          If you create the figure with `fig, fig_ax = plt.subplots()` simply pass `ax=fig_ax`.
          If you do not explicitily create a figure, then pass no extra argument.
          In this case the  current axis (i.e. `plt.gca())` will be used
    """
    if ax is None:
        ax = plt.gca()

    size = len(matrix)
    handle = plt.imshow(matrix, cmap=plt.get_cmap('summer'))
    plt.colorbar(handle)
    for i in range(size):
        for j in range(size):
            plt.text(i, j, ('{:.2f}'.format(matrix[j][i])))


# Returns k number of neighbours of the point in the array
def kNeighbours(train_set, train_labels, point, k):

    unsortedDistance = []

    for i in range(0, len(train_set)):
        euclidianDistance = LA.norm(train_set[i] - point, ord=2)
        unsortedDistance.append(euclidianDistance)

    indexSortedDistance = np.argsort(unsortedDistance)

    kNeighbours = []

    for j in range(0, k):
        kNeighbours.append(train_labels[indexSortedDistance[j]])

    return kNeighbours


def knn(train_set, train_labels, test_set, k, **kwargs):

    pred_set = []

    train_set_red, test_set_red = reduce_data(train_set, test_set, [9, 12])

    for i in range(0, len(test_set_red)):
        neighbours = kNeighbours(
            train_set_red, train_labels, test_set_red[i], k)

        labels, label_occurence = np.unique(neighbours, return_counts=True)
        predicted_labels = labels[np.argmax(label_occurence)]
        print('Predicted_labels %s', predicted_labels)
        pred_set.append(predicted_labels)

    accuracy = calculate_accuracy(test_labels, pred_set)
    print(accuracy)

    confusionMatrix = calculate_confusion_matrix(test_labels, pred_set)
    plot_matrix(confusionMatrix)
    plt.show()

    return pred_set


def alternative_classifier(train_set, train_labels, test_set, test_labels, **kwargs):
    pred_set = []

    train_set_red, test_set_red = reduce_data(train_set, test_set, [9, 12])

    train_data = np.insert(train_set_red, 2, train_labels, axis=1)
    test_data = np.insert(test_set_red, 2, test_labels, axis=1)

    tree = build_tree(train_data)

    for row in test_data:
        prediction = classify(row, tree)
        pred_set.append(prediction)

    accuracy = calculate_accuracy(test_labels, pred_set)
    print(accuracy)

    confusionMatrix = calculate_confusion_matrix(test_labels, pred_set)
    plot_matrix(confusionMatrix)
    plt.show()

    return pred_set


def knn_three_features(train_set, train_labels, test_set, k, **kwargs):
    pred_set = []

    train_set_red, test_set_red = reduce_data(train_set, test_set, [9, 12, 10])
    print(train_set_red, test_set_red)

    for i in range(0, len(test_set_red)):
        neighbours = kNeighbours(
            train_set_red, train_labels, test_set_red[i], k)

        labels, label_occurence = np.unique(neighbours, return_counts=True)
        predicted_labels = labels[np.argmax(label_occurence)]

        pred_set.append(predicted_labels)

    accuracy = calculate_accuracy(test_labels, pred_set)
    print(accuracy)

    confusionMatrix = calculate_confusion_matrix(test_labels, pred_set)
    plot_matrix(confusionMatrix)
    plt.show()

    return pred_set


def knn_pca(train_set, train_labels, test_set, k, n_components=2, **kwargs):
    # write your code here and make sure you return the predictions at the end of
    # the function
    return []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs=1, type=str,
                        help='Running mode. Must be one of the following modes: {}'.format(MODES))
    parser.add_argument('--k', nargs='?', type=int, default=1,
                        help='Number of neighbours for knn')
    parser.add_argument('--train_set_path', nargs='?', type=str,
                        default='data/wine_train.csv', help='Path to the training set csv')
    parser.add_argument('--train_labels_path', nargs='?', type=str,
                        default='data/wine_train_labels.csv', help='Path to training labels')
    parser.add_argument('--test_set_path', nargs='?', type=str,
                        default='data/wine_test.csv', help='Path to the test set csv')
    parser.add_argument('--test_labels_path', nargs='?', type=str,
                        default='data/wine_test_labels.csv', help='Path to the test labels csv')

    args = parser.parse_args()
    mode = args.mode[0]

    return args, mode


if __name__ == '__main__':
    args, mode = parse_args()  # get argument from the command line

    # load the data
    train_set, train_labels, test_set, test_labels = load_data(train_set_path=args.train_set_path,
                                                               train_labels_path=args.train_labels_path,
                                                               test_set_path=args.test_set_path,
                                                               test_labels_path=args.test_labels_path)
    if mode == 'feature_sel':
        selected_features = feature_selection(train_set, train_labels)
        print_features(selected_features)
    elif mode == 'knn':
        predictions = knn(train_set, train_labels, test_set, args.k)
        print_predictions(predictions)
    elif mode == 'alt':
        predictions = alternative_classifier(
            train_set, train_labels, test_set, test_labels)
        print_predictions(predictions)
    elif mode == 'knn_3d':
        predictions = knn_three_features(
            train_set, train_labels, test_set, args.k)
        print_predictions(predictions)
    elif mode == 'knn_pca':
        prediction = knn_pca(train_set, train_labels, test_set, args.k)
        print_predictions(prediction)
    else:
        raise Exception(
            'Unrecognised mode: {}. Possible modes are: {}'.format(mode, MODES))
