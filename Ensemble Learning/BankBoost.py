
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
#X_train = pd.get_dummies(X_train)
#print(X_train.head())
y_train = traindf["y"]
X_train = pd.get_dummies(X_train)

X_test = testdf.drop(columns="y")
#X_test = pd.get_dummies(X_test)
#print(X_test.head())
y_test = testdf["y"]

# assign our individually defined functions as methods of our classifier

clf = boost.AdaBoost().fit(X_train, y_train, iters=2)

pred = clf.predict(X)

print(pred)

train_err = (clf.predict(X) != y).mean()
print(f'Train error: {train_err:.4%}')

errors = list(clf.errors)
ada_error = list(clf.ada_errors)

y = []
for i in range(1,3):
  y.append(i)

# plotting the error curve (expected - exponentially decreasing)
plt.plot(y,errors)
plt.plot(y,ada_error)
plt.ylabel('error')
plt.xlabel('iteration')
plt.legend(['weak hypothesis error', 'final hypothesis error'], loc='upper right')
plt.show()

