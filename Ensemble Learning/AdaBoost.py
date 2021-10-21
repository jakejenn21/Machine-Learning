import numpy as np
import csv
import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("./Decision Tree")
import ID3Classifier as classifier
import numbers

#weak learner(decision stump)

class AdaBoost:

    def __init__(self):
        self.stumps = None
        self.stump_weights = None
        self.errors = None
        self.sample_weights = None
        self.ada_errors = None

    def fit(self, X: np.ndarray, y: np.ndarray, iters: int):

        n = X.shape[0]

        # initlizing numpy arrays
        self.sample_weights = np.zeros(shape=(iters, n))
        self.stumps = np.zeros(shape=iters, dtype=object)
        self.stump_weights = np.zeros(shape=iters)
        self.errors = np.zeros(shape=iters)
        self.ada_errors = np.zeros(shape=iters)

        # initializing weights uniformly
        self.sample_weights[0] = np.ones(shape=n) / n

        for t in range(iters):
            # fitting weak learner
            curr_sample_weights = self.sample_weights[t]
            stump = classifier.ID3Classifier(criterion="ig", max_depth=1, numeric_conv=True)
            stump = stump.fit(X, y, curr_sample_weights)
            stump_pred = stump.predict(X)
      # calculating error and stump weight from weak learner prediction
            err = curr_sample_weights[(stump_pred != y)].sum() / n
            if err != 0:
                stump_weight = np.log((1 - err) / err) / 2
            else:
                stump_weight = 0.5

            print(curr_sample_weights)
            print(np.exp(-stump_weight * y * stump_pred))
            # updating sample weights
            new_sample_weights = (
                float(curr_sample_weights) * float(np.exp(-stump_weight * y * stump_pred))
            )
      
            new_sample_weights /= new_sample_weights.sum()

            # updating sample weights for t+1
            if t+1 < iters:
                self.sample_weights[t+1] = new_sample_weights
  
            self.stumps[t] = stump
            self.stump_weights[t] = stump_weight
            self.errors[t] = err
            self.ada_errors[t] = np.prod(((self.errors[t]*(1-self.errors[t]))**1/2))

        return self

    def predict(self, X):
        return np.sign(np.dot(self.stump_weights, np.array([stump.predict(X) for stump in self.stumps])))