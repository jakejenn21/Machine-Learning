
import numpy as np
import csv
import pprint
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("./Decision Tree")
import ID3Classifier as classifier

class Bag:
    
    def fit(self, X_train, y_train, X_test, y_test, B, max_depth = 100, min_size = 2, seed = None):
        self.X_train = X_train
        self.N, self.D = X_train.shape
        self.y_train = y_train
        self.B = B
        self.seed = seed
        self.trees = []
        self.train_errs=[]
        self.test_errs=[]
        
        np.random.seed(seed)
        for b in range(self.B):
            
            sample = np.random.choice(np.arange(self.N), size = self.N, replace = True)
            X_train_b = X_train.iloc[sample,:]
            y_train_b = y_train.iloc[sample]
            
            y_train_b.columns="y"
            tree = classifier.ID3Classifier(criterion="ig", max_depth=max_depth, missing_value=False, sample_weights=[], numeric_conv=True, enable_categorical=True)
            tree.fit(X_train_b, y_train_b)

            pred_train = tree.predict(X_train)
            pred_test = tree.predict(X_test)

            pred_train = np.array(pred_train)
            pred_test = np.array(pred_test)
     
            y = np.array(y_train)

            # calculating error and stump weight from weak learner prediction
            stump_err_train = (pred_train != y).mean()

            y = np.array(y_test)

            # calculating error and stump weight from weak learner prediction
            stump_err_test = (pred_test != y).mean()

            y = np.array(y_test)
            print("DT err train: ", stump_err_train)
            print("DT err test: ", stump_err_test)
            stump_err_test = (pred_test != y).mean()
            self.trees.append(tree)
            self.train_errs.append(stump_err_train)
            self.test_errs.append(stump_err_test)
            
        
    def predict(self, X_test):
        y_test_hats = np.empty((len(self.trees), len(X_test)))
        for i, tree in enumerate(self.trees):
            y_pred = tree.predict(X_test)
            y_pred = np.where(y_pred=='yes', 1, -1)
            y_test_hats[i] = y_pred
        
        return y_test_hats.mean(0)