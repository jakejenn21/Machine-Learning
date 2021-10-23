
import numpy as np
import csv
import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("./Ensemble Learning")
#import ID3Classifier as classifier
#from sklearn.tree import DecisionTreeClassifier
import AdaBoost as boost
import Bag 
import numbers

print("------------------------------BANK DATASET--------------------------------\n")

attributes = {
                    'age': None, 
                    'job': ["admin.", "unknown", "unemployed", "management", "housemaid", "entrepreneur", "student",
                                       "blue-collar", "self-employed", "retired", "technician", "services"],
                    'marital': ["married", "divorced", "single"],
                    'education': ["unknown", "secondary", "primary", "tertiary"], 
                    'default': ['yes', 'no'],
                    'balance': None,
                    'housing': ['yes', 'no'],
                    'loan': ['yes', 'no'],
                    'contact': ['unknown', 'telephone', 'cellular'],
                    'day': None,
                    'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
                    'duration': None,
                    'campaign': None,
                    'pdays': None,
                    'previous': None,
                    'poutcome': ['unknown', 'other', 'failure', 'success']
            }


traindf = pd.read_csv("Decision Tree/bank/train.csv")
traindf.columns = list(attributes.keys()) + ["y"]
testdf = pd.read_csv("Decision Tree/bank/test.csv")
testdf.columns = list(attributes.keys()) + ["y"]

X_train = traindf.drop(columns="y")
#print(X_train)
#print(type(X_train))
#X_train = pd.get_dummies(X_train)
#print(X_train.head())
y_train = traindf["y"]

y_train.columns=["y"]
#y_train.columns = "y"
#print(y_train)
#X_train = pd.get_dummies(X_train)

X_test = testdf.drop(columns="y")
#X_test = pd.get_dummies(X_test)
#print(X_test.head())
y_test = testdf["y"]
#y_test.columns = "y"
num_trees = 500
training_errors=[]
test_errors=[]
x=[]
## Build model
for i in range(1, num_trees+1):

    bagger = Bag.Bag()
    bagger.fit(X_train, y_train, X_test, y_test, B = i, max_depth = 100, min_size = 5, seed = 123)
    y_train_hat = bagger.predict(X_train)
    y_train_temp = np.array(np.where(y_train=='yes', 1, -1))

    train_err = (y_train_hat != y_train_temp).mean()

    y_test_hat = bagger.predict(X_test)
    y_test_temp = np.array(np.where(y_test=='yes', 1, -1))


    test_err = (y_test_hat != y_test_temp).mean()

    training_errors.append(train_err)
    test_errors.append(test_err)
    x.append(i)

## Plot
plt.figure(1)
plt.plot(x,training_errors)
plt.plot(x,test_errors)
plt.ylabel('error')
plt.xlabel('num trees')
plt.legend(['final train error', 'final test error'], loc='upper right')
plt.show()

## Plot
plt.figure(2)
plt.plot(x,bagger.train_errs)
plt.plot(x,bagger.test_errs)
plt.ylabel('error')
plt.xlabel('num trees')
plt.legend(['weak hypothesis train error', 'weak hypothesis test error'], loc='upper right')
plt.show()