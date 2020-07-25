import time as t
import numpy as np
import copy as cp
import pickle


# Function which takes labels to give one-hot encoded vectors
def one_hot_encoder(y):
    """
    Args:
    y: labels vector
    
    returns:
    A vector of One hot encoded labels, where each one hot encoded label is 4-dimensional 
    """
    labels = y.astype(int) // 90
    label_one_hot = np.zeros((labels.size, int(labels.max() + 1)))
    label_one_hot[np.arange(labels.size), labels] = 1
    return label_one_hot.T


# Function which converts data ready for modelling from 2d numpy array of data
def model_data(data):
    """
    Args:
    data: A 2d numpy array
    
    returns:
    x: features from the data for modeling 2d array with each row having a feature vector for a data point
    y: one hot encoded labels corresponding to each x
    labels: target labels corresponding to each x which are not one-hot encoded
    """
    labels = data[:, 0].astype(int)
    x = data[:, 1:].T / 255
    y = one_hot_encoder(data[:, 0])
    return x, y, labels


# Function which dumps model weights and biases to model_file_name
def dumpTheDataToModeltxt(solution, model_file_name):
    if solution is None:
        return

    model_file = open(model_file_name, "wb+")
    pickle.dump(solution, model_file)


# Function which reads model parameters from model_file_name
def retrieveDataFromModeltxt(model_file_name):
    return pickle.load(open(model_file_name, 'rb'))


# Function which returns x passed over relu function
def relu(x):
    return x * (x > 0)


# Function which returns x passed over relu derivative
def relu_derv(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


# Function which returns yhat passed over softmax
def softmax(yhat):
    ex = np.exp(yhat)
    p = ex / np.sum(ex, axis=0)
    return p


# Function to create neural network graph
def nnet_graph(num_inputs, num_outputs, num_hidden_units):
    """
    Args:
    num_inputs: Number of inputs to the neural network
    num_outputs: Number of outputs from the neural network
    num_hidden_units: A list with number of units in each hidden layer, 
                        and length of this list corresponds to number of hidden layers
    
    returns:
    weights: List of weight matrices for each layer assigned with random values - used Xavier Initializer
    biases: List of bias vectors for each layer assigned with random values
    """
    weights = []
    biases = []

    # Weights and biases for initial layer
    weights.append(np.random.randn(num_hidden_units[0], num_inputs) * np.sqrt(1. / num_inputs))
    biases.append(np.zeros((num_hidden_units[0], 1)))

    # Weights and biases for all subsequent layers except the last
    for i in range(1, len(num_hidden_units)):
        weights.append(
            np.random.randn(num_hidden_units[i], num_hidden_units[i - 1]) * np.sqrt(1. / num_hidden_units[i - 1]))
        biases.append(np.zeros((num_hidden_units[1], 1)))

    # Weights and biases for last layer
    weights.append(np.random.randn(num_outputs, num_hidden_units[-1]) * np.sqrt(1. / num_hidden_units[-1]))
    biases.append(np.zeros((num_outputs, 1)))
    return weights, biases


# Function  which does forward propagation on data using the weights and biases
def feedforward(x, weights, biases):
    """
    Args:
    x: feature data
    weights: List of weight matrices for each layer
    biases: List of bias vectors for each layer 
    
    returns:
    weights: List of weight matrices for each layer assigned with random values - used Xavier Initializer
    biases: List of bias vectors for each layer assigned with random values
    """
    z = []
    a = []
    # First layer output and passed over relu activation
    temp = (weights[0] @ x) + biases[0]
    z.append(temp)
    temp = relu(temp)
    a.append(temp)
    temp2 = temp

    # Second layer to Last layer
    for i in range(1, len(weights)):
        temp = (weights[i] @ temp2) + biases[i]
        z.append(temp)
        if i != len(weights) - 1:
            # All layers except last layer is passed over relu activation
            temp = relu(temp)
        else:
            # Last layer passed over softmax activation
            temp = softmax(temp)
        a.append(temp)
        temp2 = temp
    return z, a


# Function which takes previous weights and biases, calculates gradients and updates the weights and biases
def backprop(x, y, wieghts_list, biases_list, z, a, lr):
    """
    Args:
    x: feature data
    y: one-hot encoded target labels
    wieghts_list: List of weight matrices for each layer from last updated
    biases_list: List of bias vectors for each layer from last updated
    z: List of layer outputs from previous feed forward process
    a: List of activations from previous feed forward process 
    
    returns:
    weights: List of weight matrices for each layer updated after backpropagation process
    biases: List of bias vectors for each layer updated after backpropagation process
    """
    weights = cp.deepcopy(wieghts_list)
    biases = cp.deepcopy(biases_list)
    dw = []
    db = []
    # last layer gradient calculations
    delta = a[-1] - y
    delta_w = lr * (delta @ a[-2].T)
    dw.append(delta_w)
    delta_b = lr * np.sum(delta, axis=1, keepdims=True)
    db.append(delta_b)

    # last but one layer to first layer gradient calculation
    for i in reversed(range(1, len(a))):
        delta = (weights[i].T @ delta) * relu_derv(z[i - 1])
        if i != 1:
            delta_w = lr * (delta @ a[i - 2].T)
        else:
            delta_w = lr * (delta @ x.T)
        delta_b = lr * np.sum(delta, axis=1, keepdims=True)

        dw.append(delta_w)
        db.append(delta_b)
    # reverse the gradient list to be in same order as weights and biases
    dw.reverse()
    db.reverse()
    # Weights and biases update
    for i in range(len(weights)):
        weights[i] = weights[i] - dw[i]
        biases[i] = biases[i] - db[i]
    return weights, biases


def model_run(x, weights, biases):
    """
    Args:
    x: feature data
    weights: List of weight matrices for each layer
    biases: List of bias vectors for each layer 
    
    returns:
    ypred: predicted labels - NOT in one-hot encoded form
    """
    _, activations = feedforward(x, weights, biases)
    ypred = np.argmax(activations[-1], axis=0) * 90
    return ypred


# Function which returns accuracy given actual 'y' and 'ypred'
def score(y, ypred):
    return (1 * (ypred == y).mean())


# Creates output file with predicted values
def output_file(test_image_ids, y_pred):
    c = [[a, b] for a, b in zip(test_image_ids, y_pred)]
    with open("output.txt", "w") as file:
        for i in range(len(c)):
            file.write(str(c[i][0]) + ' ' + str(int(c[i][1])))
            if i != len(c) - 1:
                file.write("\n")


# Important function which trains the neural network model
def train_nnet(epochs, learning_rate, hidden_units_list, train_data):
    """
    Args:
    epochs: Number of feedforward and backpropagation combinations to run
    learning_rate: Learning rate to improve the model
    hidden_units_list: A list with number of units in each hidden layer, 
                        and length of this list corresponds to number of hidden layers 
    
    returns:
    best_weights: final best weights the model has learnt from features of training data
    best_biases: final best biases the model has learnt from features of training data
    """

    # splitting the data into train and validation
    data_split_ratio = 0.8
    p = np.random.permutation(train_data.shape[0])
    train_data_shuff = train_data[p, :]
    training_data, validation_data = np.split(train_data_shuff, [int(data_split_ratio * len(train_data_shuff))])

    # getting the data ready for modelling
    trainX, trainY, train_labels = model_data(training_data)
    validX, validY, valid_labels = model_data(validation_data)

    # Model Initialization with random weights
    weights, biases = nnet_graph(trainX.shape[0], 4, hidden_units_list)
    best_weights, best_biases = cp.deepcopy(weights), cp.deepcopy(biases)
    valid_acc = score(valid_labels, model_run(validX, best_weights, best_biases))

    print('Train Accuracy, Validation Accuracy')
    # Training the weights
    for epoch in range(epochs):
        # Using stochastic gradient descent
        permutation = np.random.permutation(trainX.shape[1])
        X, Y, shuff_labels = trainX[:, permutation], trainY[:, permutation], train_labels[permutation]
        for iteration in range(X.shape[1]):
            batch_x, batch_y = X[:, iteration].reshape(X.shape[0], 1), Y[:, iteration].reshape(Y.shape[0], 1)
            # Forward and Backprop steps
            z_vals, activations = feedforward(batch_x, weights, biases)
            weights, biases = backprop(batch_x, batch_y, weights, biases, z_vals, activations, learning_rate)

        print(score(shuff_labels, model_run(X, weights, biases)),
              score(valid_labels, model_run(validX, weights, biases)))

        # Updating the best weights and biases
        if valid_acc < score(valid_labels, model_run(validX, weights, biases)):
            valid_acc = score(valid_labels, model_run(validX, weights, biases))
            best_weights, best_biases = cp.deepcopy(weights), cp.deepcopy(biases)
    return best_weights, best_biases


# Running all main train code
def train(train_data, model_file_name):
    """
    Args:
    train_data: training data before x,y split and train validation split
    model_file_name: Dumps model parameters to this file
    
    """
    start_time = t.time()
    print("Model Training...")

    epochs = 20
    learning_rate = 0.005
    model_architecture = [50, 50, 50]
    best_weights, best_biases = train_nnet(epochs, learning_rate, model_architecture, train_data)

    print("Time taken: ", (t.time() - start_time))

    # Dumping best_weights and best_biases into modelfile in dictionary format
    model_params = {}
    for i in range(len(best_weights)):
        model_params['w' + str(i)] = best_weights[i]
        model_params['b' + str(i)] = best_biases[i]
    dumpTheDataToModeltxt(model_params, model_file_name)


# Function which takes model_filename, retrieves and return best_weights, best_biases in list format
def retrieve_weights(model_file_name):
    model_params = retrieveDataFromModeltxt(model_file_name)
    best_weights = [None] * (len(model_params) // 2)
    best_biases = [None] * (len(model_params) // 2)
    for i in range(len(best_weights)):
        best_weights[i] = model_params['w' + str(i)]
        best_biases[i] = model_params['b' + str(i)]
    return best_weights, best_biases


# Given model_file_name, retrieves best_weights, best_biases and runs the model on test_data
def test(model_file_name, test_data, test_image_ids):
    best_weights, best_biases = retrieve_weights(model_file_name)
    testX, testY, test_labels = model_data(test_data)
    y_pred = model_run(testX, best_weights, best_biases)
    output_file(test_image_ids, y_pred)
    print(score(test_labels, model_run(testX, best_weights, best_biases)))
