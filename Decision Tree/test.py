import numpy as np
import csv
import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ID3Classifier as classifier

from numpy import log2 as log
from pprint import pprint

def accuracy_score(test, pred):
    return np.mean(pred == test)

#read car dataset 

traindf = pd.read_csv("Decision Tree/car/train.csv", header=None)
traindf.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']
#print(traindf)
testdf = pd.read_csv("Decision Tree/car/test.csv", header=None)
testdf.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']
#print(testdf)

# organize data into input and output
X_train = traindf.drop(columns="label")
y_train = traindf["label"]

#print(X_train)
#print(y_train)

X_test = testdf.drop(columns="label")
y_test = testdf["label"]
#print(X_test)
#print(y_test)

# model = classifier.ID3Classifier(criterion="ig", max_depth=6)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# score = accuracy_score(y_test, y_pred)
# print("ig: ", score)

# model = classifier.ID3Classifier(criterion="me", max_depth=6)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# score = accuracy_score(y_test, y_pred)
# print("me: ", score)

# model = classifier.ID3Classifier(criterion="gini", max_depth=6)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# score = accuracy_score(y_test, y_pred)
# print("gini", score)

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


traindf = pd.read_csv("Decision Tree/bank/train1.csv")
traindf.columns = list(attributes.keys()) + ["y"]
testdf = pd.read_csv("Decision Tree/bank/test1.csv")
testdf.columns = list(attributes.keys()) + ["y"]


X_train = traindf.drop(columns="y")
y_train = traindf["y"]

X_test = testdf.drop(columns="y")
y_test = testdf["y"]

model = classifier.ID3Classifier(criterion="ig", max_depth=16, numeric_conv=True)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)
print("ig: ", score)

model = classifier.ID3Classifier(criterion="me", max_depth=1, numeric_conv=True)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)
print("me: ", score)

model = classifier.ID3Classifier(criterion="gini", max_depth=1, numeric_conv=True)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)
print("gini", score)