# We will  rst implement SVM in the primal domain with stochastic sub-gradient descent. 
# We will reuse the dataset for Perceptron implementation, namely: \bank-note.zip" in Canvas. 
# The training data are stored in the  le \classification/train.csv", consisting of 872 examples. 
# The test data are stored in \classification/test.csv", and comprise of 500 examples. 
# Set the maximum epochs T to 100. Don't forget to shuffle the training examples at the start of each epoch. 
# Use the curve of the objective function (along with the number of updates) to diagnosis the convergence. 
# Try the hyperparameter given.
# Don't forget to convert the labels to be in (-1,1)

# Use the schedule of learning rate: 
# gamma_t = gamma_0/(1+gamma_0/alpha*t)

# Please tune gamma_0 and alpha to ensure convergence. 
# For each setting of C, report your training and test error.

#Use the schedule:
# gamma_t = gamma_0/(1+gamma_0/alpha*t)

#Report the training and test error for each setting of C.

# For each C:
# report the differences between the model parameters learned from the two learning rate schedules
# as well as the differences between the training/test errors. 
# What can you conclude?

import pandas as pd
import numpy as np

from sklearn.utils import shuffle

def add_regularization(w, subgradient_w):
    """
    The total loss :( 1/2 * ||w||^2 + Hingle_loss) has w term to be added after getting subgradient of 'w'
    
      total_w = regularization_term + subgradient_term
    i.e total_w = w + C *  ∑ (-y*x)
    
    """
    return w + subgradient_w

def subgradients(x, y, w, C):
    """
    :x: inputs [[x1,x2], [x2,x2],...]
    :y: labels [1, -1,...]
    :w: initial w
    :C: tradeoff/ hyperparameter
    
    """
    subgrad_w = 0
    # sum over all subgradients of hinge loss for a given samples x,y
    for x_i, y_i in zip(x,y):
        f_xi = np.dot(w.T, x_i)

        decision_value = y_i * f_xi

        if decision_value < 1:
            subgrad_w += - y_i*x_i
        else:
            subgrad_w += 0
    
    # multiply by C after summation of all subgradients for a given samples of x,y
    subgrad_w = C * subgrad_w

    #print(subgrad_w)

    return (add_regularization(w, subgrad_w))

 #Set the maximum epochs T to 100. 
 #Don't forget to shuffle the training examples at the start of each epoch. 
def primal_stochastic_SVM(data, lr_setting, gamma_0, alpha, C, T=1):
    """
    :data: Pandas data frame
    :gamma_0: 
    :alpha:
    :C: hyperparameter, tradeoff between hard margin and hinge loss
    :T: # of iterations

    """

    #numpy arrays
    X_train = data.iloc[:,:-1]
    X = X_train.to_numpy()
    w = np.zeros(X.shape[1])

    for t in range(1, T+1):
        # set learning rate
        if lr_setting == 0:
            learning_rate = gamma_0/(1 + (gamma_0/alpha)*t)
        else:
            learning_rate = gamma_0/(1 + t)

        #Shuffle Dataset
        training_sample = shuffle(data)

        #numpy arrays
        X_train = training_sample.iloc[:,:-1]

        y_train = training_sample.iloc[:,-1]

        X = X_train.to_numpy()

        y = y_train.to_numpy()

        y = np.where(y==1, 1, -1)

        # print(X)
        # print(type(X))
        #print(y)
        # print(type(y))
        
        #subgradients
        sub_grads = subgradients(X, y, w, C)

        w = w - learning_rate * sub_grads

    return w

def predict(data, w):
    #numpy arrays
    X_data = data.iloc[:,:-1]
    X_data = X_data.to_numpy()
    preds = []

    for i in range(X_data.shape[0]):
        y = np.dot(w.T,X_data[i])
        pred = np.sign(y)
        preds.append(pred)

    return preds
