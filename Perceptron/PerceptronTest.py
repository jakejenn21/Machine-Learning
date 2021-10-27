import numpy as np
import csv
import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import StandardPerceptron as model
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle


#read concrete dataset
traindf = pd.read_csv("Perceptron/bank-note/train.csv", header=None)
traindf.insert(0,'', float(1.0))
#print(traindf)
testdf = pd.read_csv("Perceptron/bank-note/test.csv", header=None)
testdf.insert(0,'', float(1.0))
#print(testdf)

model = model.StandardPerceptron(learning_rate=0.01, n_iters=5)

model.fit(traindf)

y_pred_train = []
X = np.array(traindf.iloc[:,:-1])
for i in range(0,X.shape[0]):
    y_pred_train.append(model.predict(X[i]))

#print(y_pred_train)


y_pred_test = []
X = np.array(testdf.iloc[:,:-1])
for i in range(0,X.shape[0]):
    y_pred_test.append(model.predict(X[i]))

y_train = traindf.iloc[:,-1:]

y_test = testdf.iloc[:,-1:]

#print(y_pred_test)

print("Train Accuracy: ", accuracy_score(y_train, y_pred_train))
print("Test Accuracy: ", accuracy_score(y_test, y_pred_test))



