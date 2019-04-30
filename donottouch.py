#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skeleton code for CW2 submission.
We suggest you implement your code in the provided functions
Make sure to use our print_features() and print_predictions() functions
to print your results
"""

from __future__ import print_function

from pprint import pprint
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utilities import load_data, print_features, print_predictions

from sklearn.decomposition import PCA


# you may use these colours to produce the scatter plots
CLASS_1_C = r'#3366ff'
CLASS_2_C = r'#cc3300'
CLASS_3_C = r'#ffc34d'

MODES = ['feature_sel', 'knn', 'alt', 'knn_3d', 'knn_pca']


def feature_selection(train_set, train_labels, **kwargs):

	n_features = train_set.shape[1]
	fig, ax = plt.subplots(n_features, n_features)
	plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.2, hspace=0.4)

	class_1_colour = r'#3366ff'
	class_2_colour = r'#cc3300'
	class_3_colour = r'#ffc34d'

	colours = np.zeros_like(train_labels, dtype=np.object)
	colours[train_labels == 1] = class_1_colour
	colours[train_labels == 2] = class_2_colour
	colours[train_labels == 3] = class_3_colour

	for row in range(n_features):
		for col in range(n_features):
			ax[row][col].scatter(train_set[:, row], train_set[:, col], c=colours)
			ax[row][col].set_title('Features {} vs {}'.format(row+1, col+1))

	plt.show()
	return [0, 6]


#------------------------------------------
# customly-made or copied from lab sheets


def reduce_data(train_set, test_set, selected_features):
    train_set_red = train_set[:, selected_features]
    test_set_red = test_set[:, selected_features]

    return train_set_red, test_set_red


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

    handle = ax.imshow(matrix, cmap=plt.get_cmap('summer'))
    plt.colorbar(handle)

    n_rows, n_cols = matrix.shape
    for i in range(0, n_rows):
        for j in range(0, n_cols):
            plt.text(j, i, matrix[i][j])




def calculate_confusion_matrix(gt_labels, pred_labels):

    classes = len(np.unique(gt_labels))
    length = len(gt_labels)


    result = np.zeros((classes,classes))

    for i in range(0, length):
        result[gt_labels[i]-1][pred_labels[i]-1] += 1

    for i in range(0, classes):
        result[i, :] /= sum(result[i, :])

    return result


# returns an array of neighbours
# you specify k, and  it will give back neighbours of size k which are the closest ones.

# element, we want to find it's neighbours
def neighbours(train_set, train_labels,element, k):

	#all distance from the sample to the element
	distanceSet = []

	#this is euclidian distance
	for i in range(0, len(train_set)):
		d = np.linalg.norm(train_set[i]-element, ord=2)
		distanceSet.append(d)

	#now distanceSet is the same length as data set

	#this will sort them from smallest to biggest, that way it will be neighbours.
	#It will return the index of the elements
	#arg sort gives the indices of the elements sorted
	sortedDistanceSet = np.argsort(distanceSet)
	neighbourSet = []

	#if you use the index on the train_elements it will return the label
	for j in range(0, k):
		neighbourSet.append(train_labels[sortedDistanceSet[j]])

	#gives back the array of labels that are neighbours
	return neighbourSet


def calculate_accuracy(gt_labels, pred_labels):
	#pred_labels is the result of KNN function
	#().sum is the length of that list, so that will be the number of correct guesses
    return (gt_labels==pred_labels).sum()/gt_labels.size



#----------------------------------------


def knn(train_set, train_labels, test_set, k, **kwargs):

	#predSet contains predictions to what class the test elements belong to
	predSet = []

	#used to be 0 and 6
	train_set_red, test_set_red = reduce_data(train_set, test_set, [9, 12])

	# want to run the classification for every sample that we are testing
	#take the element from the test set and compare it to the training set
	for i in range(0, len(test_set_red)):
		neighbourSet = neighbours(train_set_red, train_labels,test_set_red[i], k)
		#print(neighbourSet, "forloop")
		#bin count counts how many classes are present and in what quantity
		#argmax tells me what class appeared the most times
		#clasif is the predicted class
		clasif = np.argmax(np.bincount(neighbourSet))
		predSet.append(clasif)

	acc = calculate_accuracy(test_labels, predSet)
	print(acc)

	confusionMatrix = calculate_confusion_matrix(test_labels, predSet)
	plot_matrix(confusionMatrix)
	plt.show()

	#allAccuracies(train_set, train_labels, test_set, test_labels, k, **kwargs)

	return predSet


def allAccuracies(train_set, train_labels, test_set, test_labels, k, **kwargs):

	maxAcc = 0

	for x in range(0, 13):
		for y in range(0, 13):
			predSet = []

			train_set_red, test_set_red = reduce_data(train_set, test_set, [x, y])

			for i in range(0, len(test_set_red)):
				neighbourSet = neighbours(train_set_red, train_labels,test_set_red[i], k)

				clasif = np.argmax(np.bincount(neighbourSet))
				predSet.append(clasif)

			acc = calculate_accuracy(test_labels, predSet)
			print("Accuracy for ", x, " and ", y, " is: ", acc)

#-------------------------------------------

def alternative_classifier(train_set, train_labels, test_set, **kwargs):
    # write your code here and make sure you return the predictions at the end of
    # the function
    return []


def knn_three_features(train_set, train_labels, test_set, k, **kwargs):

	predSet = []

	train_set_red, test_set_red = reduce_data(train_set, test_set, [9, 12, 3])

	for i in range(0, len(test_set_red)):
		neighbourSet = neighbours(train_set_red, train_labels,test_set_red[i], k) # TODO - change '2' to 'k'

		clasif = np.argmax(np.bincount(neighbourSet))
		predSet.append(clasif)

	confusionMatrix = calculate_confusion_matrix(test_labels, predSet)
	plot_matrix(confusionMatrix)
	plt.show

	acc = calculate_accuracy(test_labels, predSet)
	print(acc)

	return predSet




def knn_pca(train_set, train_labels, test_set, k, n_components=2, **kwargs):

	pca = PCA(2)
	pca.fit(train_set)

	trainNew = pca.transform(train_set)
	testNew = pca.transform(test_set)


	#SCATTERING THE 2 FEATURES

	fig, ax = plt.subplots(2, 1, squeeze=False)
	plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.2, hspace=0.4)

	class_1_colour = r'#3366ff'
	class_2_colour = r'#cc3300'
	class_3_colour = r'#ffc34d'

	colours = np.zeros_like(train_labels, dtype=np.object)
	colours[train_labels == 1] = class_1_colour
	colours[train_labels == 2] = class_2_colour
	colours[train_labels == 3] = class_3_colour


	row = 0
	col = 1
	ax[0][0].scatter(trainNew[:, row], trainNew[:, col], c=colours)
	ax[0][0].set_title('Features generated by the PCA')
	ax[1][0].scatter(train_set[:, 0], train_set[:, 6], c=colours)
	ax[1][0].set_title('Features 1 and 7')


	plt.show()

	## the knn

	predSet = []

	for i in range(0, len(testNew)):
		neighbourSet = neighbours(trainNew, train_labels,testNew[i], k)

		clasif = np.argmax(np.bincount(neighbourSet))
		predSet.append(clasif)


	#confusionMatrix = calculate_confusion_matrix(test_set, predSet)
	#plot_matrix(confusionMatrix)

	#acc = calculate_accuracy(test_labels, predSet)
	#print(acc)


	return predSet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs=1, type=str, help='Running mode. Must be one of the following modes: {}'.format(MODES))
    parser.add_argument('--k', nargs='?', type=int, default=1, help='Number of neighbours for knn')
    parser.add_argument('--train_set_path', nargs='?', type=str, default='data/wine_train.csv', help='Path to the training set csv')
    parser.add_argument('--train_labels_path', nargs='?', type=str, default='data/wine_train_labels.csv', help='Path to training labels')
    parser.add_argument('--test_set_path', nargs='?', type=str, default='data/wine_test.csv', help='Path to the test set csv')
    parser.add_argument('--test_labels_path', nargs='?', type=str, default='data/wine_test_labels.csv', help='Path to the test labels csv')

    args = parser.parse_args()
    mode = args.mode[0]

    return args, mode


if __name__ == '__main__':
    args, mode = parse_args() # get argument from the command line

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
        predictions = alternative_classifier(train_set, train_labels, test_set)
        print_predictions(predictions)
    elif mode == 'knn_3d':
        predictions = knn_three_features(train_set, train_labels, test_set, args.k)
        print_predictions(predictions)
    elif mode == 'knn_pca':
        prediction = knn_pca(train_set, train_labels, test_set, args.k)
        print_predictions(prediction)
    else:
        raise Exception('Unrecognised mode: {}. Possible modes are: {}'.format(mode, MODES))
