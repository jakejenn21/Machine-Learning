
import numpy as np
import csv
import pprint
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import sys
sys.path.append("./Ensemble Learning")
import AdaBoost as boost
import numbers

#print("------------------------------BANK DATASET--------------------------------\n")

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
y_train = traindf["y"]

y_train.columns=["y"]

X_test = testdf.drop(columns="y")
y_test = testdf["y"]

iters = 500

training_errors=[]
test_errors=[]
x = []

for i in range(1,iters+1):
  #print(i)
  clf = boost.AdaBoost().fit(X_train, y_train, X_test, y_test, i)

  pred_train = clf.predict(X_train)
  
  y_train_temp = np.array(y_train)
  y_train_temp = np.where(y_train_temp=='yes', 1, -1)

  train_err = (pred_train != y_train_temp).mean()
  #print(f'Train error: {train_err:.1%}')

  pred_test = clf.predict(X_train)
  
  y_test_temp = np.array(y_test)
  y_test_temp = np.where(y_test_temp=='yes', 1, -1)

  test_err = (pred_test != y_test_temp).mean()
  #print(f'Test error: {test_err:.1%}')

  training_errors.append(train_err)
  test_errors.append(test_err)
  x.append(i)

# plotting the error curve (expected - exponentially decreasing)

#The first figgure shows how the training and test errors vary along with T.
plt.figure(1)
plt.plot(x,training_errors)
plt.plot(x,test_errors)
plt.title("Train vs Test Error")
plt.ylabel('error')
plt.xlabel('iteration')
plt.legend(['Training Errors', 'Test Errors'], loc='upper right')

#The second figure shows the training and test errors of all the decision stumps learned in each iteration.
plt.figure(2)
plt.plot(x,clf.stump_errors_train)
plt.plot(x,clf.stump_errors_test)
plt.title("Train vs Test Error of Decision Stumps Learned in each iteration")
plt.ylabel('error')
plt.xlabel('iteration')
plt.legend(['Train Error of Decision Stump', 'Test Error of Decision Stump'], loc='upper right')
plt.show()

