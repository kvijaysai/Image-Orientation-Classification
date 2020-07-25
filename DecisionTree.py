#!/usr/local/bin/python3
###################################
# CS B551 Fall 2019, Assignment #4
#
#
import time as t
import numpy as np
import copy as cp
import math
import json
import pickle


#Calculates acuracy of the model
def accuracy_metric(y_act, y_pred):
    correct = (y_act == y_pred).sum()
    metric = correct / len(y_act)
    return metric * 100

# Calculates entropy for a class
def entopy_fun(nic, nc, base):
    return -(nic / nc) * math.log(nic / nc, base)


def entropy(X, Y, Y_Class, A,thre):
    """
    Args:
    X: subset of Train data (filtered data based decision path followed till this node )
    Y: subset of target variable in Train data (filtered data based decision path followed till this node )
    Y_class: Nnumber of output classes ( here it is 4 (0,90,180,270))
    A: Attribute column ID for which Entropy is calculated
    
    returns:
    e = Entropy corresponding to a split on an Attribute
    """
    s_left = 0;
    s_right = 0;
    y_pred = X[:, A] < thre
    Y_left = Y[y_pred]
    Y_right = Y[~y_pred]
    n_class = len(Y_Class)
    # Looping to calculate entopies due to each class
    for j in Y_Class:
        y_left_class = Y_left[Y_left == j]
        y_right_class = Y_right[Y_right == j]
        if len(y_left_class) > 0:
            s_left += entopy_fun(len(y_left_class), len(Y_left), n_class)
        if len(y_right_class) > 0:
            s_right += entopy_fun(len(y_right_class), len(Y_right), n_class)
    e = (len(Y_left) / len(Y)) * (s_left) + (len(Y_right) / len(Y)) * (s_right)
    return e

#Function to decide best split on an attribute i.e best threshold
def best_split_one(X, Y, Y_Class, A,sample_count):
    """
    Args:
    X: subset of Train data (filtered data based decision path followed till this node )
    Y: subset of target variable in Train data (filtered data based decision path followed till this node )
    Y_class: Nnumber of output classes ( here it is 4 (0,90,180,270))
    A: Attribute column ID for which Entropy is calculated
    sample_count: Number of random numbers to find best threshold
    returns:
    min_e = Entropy corresponding to a best split on an Attribute
    opt_thre = Optimal cutoff (i.e cutoff with minimum entropy)
    """
    X_unique = np.random.uniform(0, 255, sample_count)
    min_e = 100
    opt_thre = 0
    for i in X_unique:
        e = entropy(X, Y, Y_Class, A, i)
        if e <= min_e:
            min_e = e
            opt_thre = i
    return min_e, opt_thre

#Finds best attribute and corresponding threshold for a node in tree
def best_split_all(X, Y, Y_Class,sample_count, used_col, Attributes):
    """
    Args:
    X: subset of Train data (filtered data based decision path followed till this node )
    Y: subset of target variable in Train data (filtered data based decision path followed till this node )
    Y_class: Nnumber of output classes ( here it is 4 (0,90,180,270))
    sample_count: Number of random numbers to find best threshold
    used_col: Attributes that already present in the path(these won't be used again)
    Attribute: List of total Attributes available( whole features)
    returns:
    Att_col: Best attribute column number
    opt_thre = Optimal cutoff (i.e cutoff with minimum entropy)
    min_e = Entropy corresponding to a best split on an Attribute
    """
    min_e = 100
    opt_thre = 0
    Att_col = 0
    for i in Attributes:
        if i not in used_col:
            e, thre = best_split_one(X, Y, Y_Class, i,sample_count)
            if e == 0:
                return e, thre, i
            elif e <= min_e:
                min_e = e
                opt_thre = thre
                Att_col = i
    return Att_col, opt_thre, min_e

#Function to check whether all values in a 'y' as same
def all_same(y):
    """
    Args:
    y: 1D array with cclasses
    Return:
    Boolean values ( if all same TRUE else FALSE)
    """
    return all(x == y[0] for x in y)

# # Bulids complete Tree recursively
def fit(Attributes, x, y, Y_Class,sample_count, sub_tree, cur_depth, max_depth, used_col={}, val=2):
    """
    Args:
    Attribute: List of total Attributes available( whole features)
    x: subset of Train data (filtered data based decision path followed till this node )
    y: subset of target variable in Train data (filtered data based decision path followed till this node )
    Y_class: Nnumber of output classes ( here it is 4 (0,90,180,270))
    sample_count: Number of random numbers to find best threshold
    sub_tree: Part of a decision tree that built in this recursion
    cur_depth: Current Depth in the Tree
    max_depth: Max depth of the tree needed( input parameter)
    used_col: Attributes that already present in the path(these won't be used again)
    val: Class correponding to the branch 
    returns:
    Sub_tree = fully constructed decision Tree
    """
    # if there is no data to split further, truncates branch and assigns class value to it
    if len(y) == 0:
        return {'val': val}
    #This helps to limit the depth of the tree and assigns class value with high freq 
    elif cur_depth > max_depth:
        aa = np.unique(y, return_counts=True)
        return {'val': aa[0][np.where(aa[1] == np.amax(aa[1]))][0]}
    #if all the classes are same then it truncates the branch and assigns Class value
    elif all_same(y):
        return {'val': y[0]}

    else:
        # finds out best split and attribute ( max information gain i.e min entropy)
        col, cutoff, entropy_min = best_split_all(x, y, Y_Class,sample_count,used_col, Attributes)  
        used_col_new = cp.deepcopy(used_col)
        col = int(col)
        used_col_new[col] = 1
        y_left = y[x[:, col] < cutoff]
        y_right = y[x[:, col] >= cutoff]
        aa = np.unique(y, return_counts=True)
        val = aa[0][np.where(aa[1] == np.amax(aa[1]))][0]
        sub_tree = {'col': col, 'index_col': col,
                    'cutoff': cutoff,
                    'val': val,
                    'entropy': entropy_min}
        # Bulids leftside of tree recursively
        sub_tree['left'] = fit(Attributes, x[x[:, col] < cutoff], y_left, Y_Class,sample_count, {}, cur_depth + 1, max_depth,
                               used_col_new, val)
        # Bulids rightside of tree recursively
        sub_tree['right'] = fit(Attributes, x[x[:, col] >= cutoff], y_right, Y_Class,sample_count, {}, cur_depth + 1, max_depth,
                                used_col_new, val)
        cur_depth += 1
        return sub_tree

# Function to predict the class for a test example
def predict_class(trees, x_test):
    """
    Args:
    trees: Tree that is built using training data
    x_test: Test Example
    returns:
    class values for the example
    """
    tree_level = cp.deepcopy(trees)   # Tree Built during training
    while tree_level.get('cutoff'):  # runs till the last node
        if x_test[tree_level['index_col']] < tree_level['cutoff']:  
            # Decides the direction based on Threshold for each attribute
            tree_level = tree_level['left']
        else:
            tree_level = tree_level['right']
    else:
        # if it is a leaf node, then returns the class values for the example
        return tree_level.get('val')
    
# Function to predict the classes for a all test example
def predict(trees, x_test):
    """
    Args:
    trees: Tree that is built using training data
    x_test: Test set
    returns:
    class values for all the examples
    """
    results = np.array([0] * len(x_test))
    for i, c in enumerate(x_test):  # for each row in test data
        results[i] = predict_class(trees, c)
    return results



# Creates output file with predicted values
def output_file(test_image_ids, y_pred):
    c = [[a,b] for a, b in zip(test_image_ids, y_pred)] 
    with open("output.txt", "w") as file:
        for i in range(len(c)):
            file.write(str(c[i][0]) + ' ' +str(int(c[i][1])))
            if i!=len(c)-1:
                file.write("\n")

# Level order traversal
def dumpTheDataToModeltxt(solution, model_file_name):
    if solution is None:
        return

    model_file = open(model_file_name, "wb+")
    pickle.dump(solution, model_file)


def retrieveDataFromModeltxt(model_file_name):
    return pickle.load(open(model_file_name, 'rb'))


def train(train_data, model_file_name):
    start_time = t.time()

    X_train = train_data[:, 1:]
    Y_train = train_data[:, 0]

    # Number of Attributes
    m = X_train.shape[1]
    n = X_train.shape[0]
    Y_Class = np.unique(Y_train)
    Attributes = [i for i in range(m)]

    solution = fit(Attributes=Attributes, x=X_train, y=Y_train, Y_Class=Y_Class,sample_count=5, sub_tree={}, cur_depth=0, max_depth=9)

    dumpTheDataToModeltxt(solution, model_file_name)
    # Solution should be exported to model_DT


# Below code should for testing
def test(test_data, model_file_name,test_image_ids):
    X_test = test_data[:, 1:]
    Y_test = test_data[:, 0]

    solution = retrieveDataFromModeltxt(model_file_name)
    y_pred = predict(solution, X_test)

    score_test = accuracy_metric(Y_test, y_pred)

    print(score_test)
    output_file(test_image_ids, y_pred)
