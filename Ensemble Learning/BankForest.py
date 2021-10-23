
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
import RandomForest 
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

    model = RandomForest.RandomForest()
    model.fit(X_train, y_train)
    pred_test = model.predict(X_train)

    print(pred_test)

    print(y_test)



    train_err = (pred_test != y_test).mean()

    print(train_err)

    model.fit(X_test, y_test)
    preds = model.predict(X_train)

    test_err = (preds != y_train).mean()

    print(test_err)

    training_errors.append(train_err)
    test_errors.append(test_err)
    x.append(i)

## Plot
plt.figure(1)
plt.plot(x,training_errors)
plt.plot(x,test_errors)
plt.ylabel('error')
plt.xlabel('iteration')
plt.legend(['weak hypothesis error', 'final hypothesis error'], loc='upper right')
plt.show()