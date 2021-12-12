import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score


def sigmoid(z):
    return 1.0/(1.0+np.exp(-1.0*z))


def initialize_params(layer_sizes, zero_weights, test):
    params = {}

    if test == True:
        params['W' + str(1)] = [-1.0,1.0,-2.0,2.0,-3.0,3.0]
        params['B' + str(1)] = [1.0,1.0,1.0,1.0,1.0,1.0]

        params['W' + str(2)] = [-1.0,1,0,-2.0,2.0,-3.0,3.0]
        params['B' + str(2)] = [1.0,1.0,1.0,1.0,1.0,1.0]

        params['W' + str(3)] = [-1.0,2.0,-1.5]
        params['B' + str(3)] = [1.0,1.0,1.0]

    if zero_weights == True:
        for i in range(1, len(layer_sizes)):
            params['W' + str(i)] = np.zeros((layer_sizes[i],
                                               layer_sizes[i-1]), dtype=float)
            params['B' + str(i)] = np.zeros((layer_sizes[i], 1), dtype=float)
        return params
    
    else:
        for i in range(1, len(layer_sizes)):
            params['W' + str(i)] = np.random.randn(layer_sizes[i],
                                               layer_sizes[i-1])
            params['B' + str(i)] = np.random.randn(layer_sizes[i], 1)
        return params


def forward_propagation(X_train, params):
    layers = len(params)//2
    values = {}
    for i in range(1, layers+1):
        if i == 1:
            values['Z' + str(i)] = np.dot(params['W' + str(i)],
                                          X_train) + params['B' + str(i)]
            values['A' + str(i)] = sigmoid(values['Z' + str(i)])
        else:
            values['Z' + str(i)] = np.dot(params['W' + str(i)],
                                          values['A' + str(i-1)]) + params['B' + str(i)]
            if i == layers:
                values['A' + str(i)] = values['Z' + str(i)]
            else:
                values['A' + str(i)] = sigmoid(values['Z' + str(i)])
    return values


def compute_cost(values, Y_train):
    layers = len(values)//2
    Y_pred = values['A' + str(layers)]
    Y_pred = np.where(Y_pred > 0, 1.0, 0.0)
    cost = 1/(2*len(Y_train)) * np.sum(np.square(Y_pred - Y_train))
    return cost


def backward_propagation(params, values, X_train, Y_train, test=False):

    layers = len(params)//2
    m = len(Y_train)
    grads = {}
    for i in range(layers, 0, -1):
        if i == layers:
            dA = 1/m * (values['A' + str(i)] - Y_train)
            dZ = dA
        else:
            dA = np.dot(params['W' + str(i+1)].T, dZ)
            dZ = np.multiply(dA, np.where(values['A' + str(i)] >= 0, 1, 0))
        if i == 1:
            if test == True:
                grads['W' + str(i)] = 1/m * np.dot(dZ, X_train)
            else:
                grads['W' + str(i)] = 1/m * np.dot(dZ, X_train.T)
                grads['B' + str(i)] = 1/m * np.sum(dZ, axis=1, keepdims=True)
        else:
            grads['W' + str(i)] = 1/m * np.dot(dZ, values['A' + str(i-1)].T)
            grads['B' + str(i)] = 1/m * np.sum(dZ, axis=1, keepdims=True)
    return grads


def update_params(params, grads, learning_rate):
    layers = len(params)//2
    params_updated = {}
    for i in range(1, layers+1):
        params_updated['W' + str(i)] = params['W' + str(i)] - \
            learning_rate * grads['W' + str(i)]
        params_updated['B' + str(i)] = params['B' + str(i)] - \
            learning_rate * grads['B' + str(i)]
    return params_updated


def model(X_train, Y_train, layer_sizes=[5,12,12,1], num_iters=100, gamma_0= 0.01, d=0.5, zero_weights=False, test=False):
    params = initialize_params(layer_sizes, zero_weights, test)
    for t in range(num_iters):
        if test == True:
            values = forward_propagation(X_train, params)
            #cost = compute_cost(values, Y_train)
            grads = backward_propagation(params, values, X_train, Y_train, test)
            return grads
        values = forward_propagation(X_train.T, params)
        #cost = compute_cost(values, Y_train.T)
        grads = backward_propagation(params, values, X_train.T, Y_train.T)
        learning_rate = gamma_0/(1 + (gamma_0/d)*t)
        params = update_params(params, grads, learning_rate)
        #print('Cost at iteration ' + str(t+1) + ' = ' + str(cost) + '\n')
    return params


def compute_accuracy(X_train, X_test, Y_train, Y_test, params, layer_sizes=None):
    values_train = forward_propagation(X_train.T, params)

    values_test = forward_propagation(X_test.T, params)

    values_train = values_train['A' + str(len(layer_sizes)-1)]

    values_train = np.where(values_train > 0.0, 1.0, 0.0)

    values_test = values_test['A' + str(len(layer_sizes)-1)]

    values_test = np.where(values_test > 0.0, 1.0, 0.0)

    mse_train_acc = np.sqrt(mean_squared_error(Y_train, values_train.T))
    mse_test_acc = np.sqrt(mean_squared_error(Y_test, values_test.T))

    train_acc = accuracy_score(Y_train, values_train.T)

    test_acc = accuracy_score(Y_test, values_test.T)


    return mse_train_acc, mse_test_acc, train_acc, test_acc


def predict(X, params):
    values = forward_propagation(X.T, params)
    predictions = values['A' + str(len(values)//2)].T
    return predictions


# testing

#testing for problem 3

print("\n\nTesting for Problem 3: \n")
# train the model
test_grads = model([1,1,1], [1], [3,3,3,1], 1, test=True)

print(test_grads)

print("\n")

# load dataset

# read bank note dataset
traindf = pd.read_csv("Neural Networks/bank-note/train.csv", header=None)
traindf.insert(0, '', float(1.0))
# print(traindf)
testdf = pd.read_csv("Neural Networks/bank-note/test.csv", header=None)
testdf.insert(0, '', float(1.0))
# print(testdf)

# separate data into input and output features

# numpy arrays
X_train = traindf.iloc[:, :-1]

y_train = traindf.iloc[:, -1]

y_train = np.where(y_train == 1, 1.0, 0.0)

X_test = testdf.iloc[:, :-1]

y_test = testdf.iloc[:, -1]

y_test = np.where(y_test == 1, 1.0, 0.0)


# set layer sizes, do not change the size of the first and last layer

layer_size5 = [5, 5, 5, 1]
layer_size10 = [5, 10, 10, 1]
layer_size25 = [5, 25, 25, 1]
layer_size50 = [5, 50, 50, 1]
layer_size100 = [5, 100, 100, 1]
# set number of iterations over the training set(also known as epochs in batch gradient descent context)
num_iters = 500

# set params for gradient descent
gamma_0 = 0.5
d = 0.001


#random weights

print("\n")
print("\n")
print("Random Weights")
print("\n")

# train the model
params = model(X_train, y_train, layer_size5, num_iters, gamma_0, d)
# get training and test accuracy
mse_train_acc, mse_test_acc, train_acc, test_acc = compute_accuracy(
    X_train, X_test, y_train, y_test, params, layer_size5)

print("\n")

# print results
print('Root Mean Squared Error on Training Data (width=5) = ', str(mse_train_acc))
print('Root Mean Squared Error on Test Data  (width=5) = ' + str(mse_test_acc))
print('Training Data Accuracy(width=5) = ', str(train_acc))
print('Test Data Accuracy(width=5) = ' + str(test_acc))

print("\n")

# train the model
params = model(X_train, y_train, layer_size10, num_iters, gamma_0, d)
# get training and test accuracy
mse_train_acc, mse_test_acc, train_acc, test_acc = compute_accuracy(
    X_train, X_test, y_train, y_test, params, layer_size10)

# print results
print('Root Mean Squared Error on Training Data (width=10) = ', str(mse_train_acc))
print('Root Mean Squared Error on Test Data  (width=10) = ' + str(mse_test_acc))
print('Training Data Accuracy(width=10) = ', str(train_acc))
print('Test Data Accuracy(width=10) = ' + str(test_acc))

print("\n")

# train the model
params = model(X_train, y_train, layer_size25, num_iters, gamma_0, d)
# get training and test accuracy
mse_train_acc, mse_test_acc, train_acc, test_acc = compute_accuracy(
    X_train, X_test, y_train, y_test, params, layer_size25)

# print results
print('Root Mean Squared Error on Training Data (width=25) = ', str(mse_train_acc))
print('Root Mean Squared Error on Test Data  (width=25) = ' + str(mse_test_acc))
print('Training Data Accuracy(width=25) = ', str(train_acc))
print('Test Data Accuracy(width=25) = ' + str(test_acc))

print("\n")

# train the model
params = model(X_train, y_train, layer_size50, num_iters, gamma_0, d)
# get training and test accuracy
mse_train_acc, mse_test_acc, train_acc, test_acc = compute_accuracy(
    X_train, X_test, y_train, y_test, params, layer_size50)

# print results
print('Root Mean Squared Error on Training Data (width=50) = ', str(mse_train_acc))
print('Root Mean Squared Error on Test Data  (width=50) = ' + str(mse_test_acc))
print('Training Data Accuracy(width=50) = ', str(train_acc))
print('Test Data Accuracy(width=50) = ' + str(test_acc))

print("\n")

# train the model
params = model(X_train, y_train, layer_size100, num_iters, gamma_0, d)
# get training and test accuracy
mse_train_acc, mse_test_acc, train_acc, test_acc = compute_accuracy(
    X_train, X_test, y_train, y_test, params, layer_size100)

# print results
print('Root Mean Squared Error on Training Data (width=100) = ', str(mse_train_acc))
print('Root Mean Squared Error on Test Data  (width=100) = ' + str(mse_test_acc))
print('Training Data Accuracy(width=100) = ', str(train_acc))
print('Test Data Accuracy(width=100) = ' + str(test_acc))

print("\n\n")


#zero weights

print("\n\n")
print("Zero Weights")
print("\n")

# train the model
params = model(X_train, y_train, layer_size5, num_iters, gamma_0, d, zero_weights=True)
# get training and test accuracy
mse_train_acc, mse_test_acc, train_acc, test_acc = compute_accuracy(
    X_train, X_test, y_train, y_test, params, layer_size5)

print("\n")

# print results
print('Root Mean Squared Error on Training Data (width=5) = ', str(mse_train_acc))
print('Root Mean Squared Error on Test Data  (width=5) = ' + str(mse_test_acc))
print('Training Data Accuracy(width=5) = ', str(train_acc))
print('Test Data Accuracy(width=5) = ' + str(test_acc))

print("\n")

# train the model
params = model(X_train, y_train, layer_size10, num_iters, gamma_0, d, zero_weights=True)
# get training and test accuracy
mse_train_acc, mse_test_acc, train_acc, test_acc = compute_accuracy(
    X_train, X_test, y_train, y_test, params, layer_size10)

# print results
print('Root Mean Squared Error on Training Data (width=10) = ', str(mse_train_acc))
print('Root Mean Squared Error on Test Data  (width=10) = ' + str(mse_test_acc))
print('Training Data Accuracy(width=10) = ', str(train_acc))
print('Test Data Accuracy(width=10) = ' + str(test_acc))

print("\n")

# train the model
params = model(X_train, y_train, layer_size25, num_iters, gamma_0, d, zero_weights=True)
# get training and test accuracy
mse_train_acc, mse_test_acc, train_acc, test_acc = compute_accuracy(
    X_train, X_test, y_train, y_test, params, layer_size25)

# print results
print('Root Mean Squared Error on Training Data (width=25) = ', str(mse_train_acc))
print('Root Mean Squared Error on Test Data  (width=25) = ' + str(mse_test_acc))
print('Training Data Accuracy(width=25) = ', str(train_acc))
print('Test Data Accuracy(width=25) = ' + str(test_acc))
print("\n")

# train the model
params = model(X_train, y_train, layer_size50, num_iters, gamma_0, d, zero_weights=True)
# get training and test accuracy
mse_train_acc, mse_test_acc, train_acc, test_acc = compute_accuracy(
    X_train, X_test, y_train, y_test, params, layer_size50)

# print results
print('Root Mean Squared Error on Training Data (width=50) = ', str(mse_train_acc))
print('Root Mean Squared Error on Test Data  (width=50) = ' + str(mse_test_acc))
print('Training Data Accuracy(width=50) = ', str(train_acc))
print('Test Data Accuracy(width=50) = ' + str(test_acc))

print("\n")

# train the model
params = model(X_train, y_train, layer_size100, num_iters, gamma_0, d, zero_weights=True)
# get training and test accuracy
mse_train_acc, mse_test_acc, train_acc, test_acc = compute_accuracy(
    X_train, X_test, y_train, y_test, params, layer_size100)

# print results
print('Root Mean Squared Error on Training Data (width=100) = ', str(mse_train_acc))
print('Root Mean Squared Error on Test Data  (width=100) = ' + str(mse_test_acc))
print('Training Data Accuracy(width=100) = ', str(train_acc))
print('Test Data Accuracy(width=100) = ' + str(test_acc))

print("\n\n")
