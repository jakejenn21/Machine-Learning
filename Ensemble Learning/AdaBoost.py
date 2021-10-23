import numpy as np
import csv
import pprint
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import sys
sys.path.append("./Decision Tree")
import ID3Classifier as classifier
import numbers
# Import helper functions
from sklearn.metrics import accuracy_score

#weak learner(decision stump)

def sign(x):
    return abs(x)/x if x!=0 else 1 

class AdaBoost:

    def __init__(self):
        self.stumps = None
        self.stump_weights = None
        self.stump_errors_train = None
        self.stump_errors_test = None
        self.sample_weights = None

    def fit(self, X_train, y_train, X_test, y_test, iters: int):

        num_rows, num_cols = np.array(X_test.values).shape
    
        # initlizing numpy arrays
        self.sample_weights = np.zeros(shape=(iters, num_rows))
        self.stumps = np.zeros(shape=iters, dtype=object)
        self.stump_weights = np.zeros(shape=iters)
        self.stump_errors_train = np.zeros(shape=iters)
        self.stump_errors_test = np.zeros(shape=iters)

        # initializing weights uniformly
        self.sample_weights[0] = np.ones(shape=num_rows) / num_rows

        for t in range(iters):
            min_error = float('inf')

            # fitting weak learner
            curr_sample_weights = self.sample_weights[t]
            stump = classifier.ID3Classifier(criterion="ig", max_depth=1, missing_value=False, sample_weights=[], numeric_conv=True, enable_categorical=True)
            stump.fit(X_test, y_test, curr_sample_weights)

            stump_pred_train = stump.predict(X_train)
            stump_pred_test = stump.predict(X_test)

            stump_pred_train = np.array(stump_pred_train)
            stump_pred_test = np.array(stump_pred_test)
     
            y = np.array(y_train)

            # calculating error and stump weight from weak learner prediction
            stump_err_train = np.sum(curr_sample_weights[(stump_pred_train != y)])

            y = np.array(y_test)

            stump_err_test = np.sum(curr_sample_weights[(stump_pred_test != y)])

            #print(err)
            alpha = 0.5 * math.log((1.0 - stump_err_train) / (stump_err_train + 1e-10))
            #print(stump_weight)
            stump_pred = np.where(stump_pred_train=='yes', 1, -1)
            y_temp = np.array(np.where(y=='yes', 1, -1))

            exp = np.exp(-alpha* y_temp * stump_pred)
            # updating sample weights
            new_sample_weights = (
                curr_sample_weights * exp
            )

            new_sample_weights /= np.sum(new_sample_weights)
            # updating sample weights for t+1
            if t+1 < iters:
                self.sample_weights[t+1] = new_sample_weights
    
            #print("stump train err: ", stump_err_train)
            #print("stump test err: ", stump_err_test)
            self.stumps[t] = stump
            self.stump_weights[t] = alpha
            self.stump_errors_train[t] = stump_err_train
            self.stump_errors_test[t] = stump_err_test

        return self

    def predict(self, X):
        y = 0
        for m in range(len(self.stumps)):
            model = self.stumps[m]
            alpha = self.stump_weights[m]
            y += alpha*np.where(model.predict(X)=='yes', 1, -1)
        signA = np.vectorize(sign)
        y = np.where(signA(y)==-1,-1,1)
        return y