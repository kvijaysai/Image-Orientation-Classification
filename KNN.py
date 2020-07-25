#!/usr/local/bin/python3
###################################
# CS B551 Fall 2019, Assignment #4
#
#
import time as t
import numpy as np


# Finds k nearest neighbours in train data for a test example using Euclidean distance
def KNN_neighbors(train, test_row, k):
    """
    Args:
    train: Train data with first column has target variable and rest columns has features for all examples
    test_row: List with target variable and features of a test example
    K: Number of neighbours
    
    returns:
    A class with highest freq 
    """
    # Calculates euclidean distance between test exmaple and all training data points
    # And stores a np array as [class,distance] for all points
    distances = np.column_stack([train[:, 0], np.sqrt(np.sum(np.square(train[:, 1:] - test_row[1:]), axis=1))])

    # Sorts based on distances and stores classes of k nearest points
    neighbors = list(distances[distances[:, 1].argsort()][:k:, 0])

    return max(set(neighbors), key=neighbors.count)


# Function KNN
def KNN_predict(train_data, test_data, k):
    """
    Args:
    train_data: Train data with first column has target variable and rest columns has features for all examples
    test_data: Test data with first column has target variable and rest columns has features for all examples
    K: Number of neighbours
    
    returns:
    Predicted classes for all test points
    """
    y_pred = [KNN_neighbors(train_data, y, k) for y in test_data]

    return y_pred


# Calculates Accuracy
def accuracy_metric(y_act, y_pred):
    correct = (np.array(y_act) == np.array(y_pred)).sum()
    metric = correct / len(y_act)
    return metric * 100


# Creates output file with predicted values
def output_file(test_image_ids, y_pred):
    c = [[a, b] for a, b in zip(test_image_ids, y_pred)]
    with open("output.txt", "w") as file:
        for i in range(len(c)):
            file.write(str(c[i][0]) + ' ' + str(int(c[i][1])))
            if i != len(c) - 1:
                file.write("\n")


####################
# Main program
#
def start(train_data, test_data, test_image_ids):
    start_time = t.time()

    y_pred = KNN_predict(train_data, test_data, k=10)

    Y_test = test_data[:, 0]
    score = accuracy_metric(Y_test, y_pred)
    print(score)
    output_file(test_image_ids, y_pred)


def train_data(train_data_filename, model_file_name):
    model_file = open(model_file_name, "w+")
    file = open(train_data_filename, 'r')
    for line in file:
        model_file.write(line)
