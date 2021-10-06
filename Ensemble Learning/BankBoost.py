
import numpy as np
import csv
import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("./Decision Tree")
import ID3Classifier as classifier
import AdaBoost as boost
import numbers

def make_binary(attributes, df, yes, no):
    df['age'] = np.where(df['age'] >= attributes['age'], yes, no)

    df['balance'] = np.where(df['balance'] >= attributes['balance'], yes, no)

    df['day'] = np.where(df['day'] >= attributes['day'], yes, no)

    df['duration'] = np.where(df['duration'] >= attributes['duration'], yes, no)

    df['campaign'] = np.where(df['campaign'] >= attributes['campaign'], yes, no)

    df['pdays'] = np.where(df['pdays'] >= attributes['pdays'], yes, no)

    df['previous'] = np.where(df['previous'] >= attributes['previous'], yes, no)


def accuracy_score(test, pred):
    return np.mean(pred == test)

#Prepare data for Decision Trees
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

traindf = pd.read_csv("Ensemble Learning/bank/train.csv", header=None)
traindf.columns = list(attributes.keys()) + ["label"]
#print(traindf)
testdf = pd.read_csv("Ensemble Learning/bank/test.csv", header=None)
testdf.columns = list(attributes.keys()) + ["label"]
#print(testdf)

attributes['age'] = np.median(traindf['age'])
attributes['balance'] = np.median(traindf['balance'])
attributes['day'] = np.median(traindf['day'])
attributes['duration'] = np.median(traindf['duration'])
attributes['campaign'] = np.median(traindf['campaign'])
attributes['pdays'] = np.median(traindf['pdays'])
attributes['previous'] = np.median(traindf['previous'])

#print(attributes)

make_binary(attributes, traindf, 1, -1)
make_binary(attributes, testdf, 1, -1)

#print(traindf)
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

# Fit a simple decision tree first
tree = classifier.ID3Classifier(1, 0)
tree.fit(X_train, y_train)
er_tree = boost.generic_clf(y_train, X_train, y_test, X_test, tree)

# Fit Adaboost classifier using a decision tree as base estimator
# Test with different number of iterations
er_train, er_test = [er_tree[0]], [er_tree[1]]
x_range = range(10, 410, 10)
for i in x_range:    
    er_i = boost.adaboost_clf(y_train, X_train, y_test, X_test, i, tree)
    er_train.append(er_i[0])
    er_test.append(er_i[1])
    
# Compare error rate vs number of iterations
plot_error_rate(er_train, er_test)

