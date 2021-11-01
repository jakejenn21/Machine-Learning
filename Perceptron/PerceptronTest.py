import numpy as np
import csv
import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import StandardPerceptron as model1
import VotedPerceptron as model2
import AveragedPerceptron as model3
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle


#read concrete dataset
traindf = pd.read_csv("Perceptron/bank-note/train.csv", header=None)
traindf.insert(0,'', float(1.0))
#print(traindf)
testdf = pd.read_csv("Perceptron/bank-note/test.csv", header=None)
testdf.insert(0,'', float(1.0))
#print(testdf)

print("\n\n---------------------------------Standard Perceptron---------------------------------\n")

model1 = model1.StandardPerceptron(learning_rate=0.01, n_iters=10)

model1.fit(traindf)

y_pred_train = []
X = np.array(traindf.iloc[:,:-1])
for i in range(0,X.shape[0]):
    y_pred_train.append(model1.predict(X[i]))

#print(y_pred_train)


y_pred_test = []
X = np.array(testdf.iloc[:,:-1])
for i in range(0,X.shape[0]):
    y_pred_test.append(model1.predict(X[i]))

y_train = traindf.iloc[:,-1:]

y_test = testdf.iloc[:,-1:]

#print(y_pred_test)
#print("\n\nTrain Accuracy: ", accuracy_score(y_train, y_pred_train), "\n\n")
print("\n\nTest Accuracy: ", accuracy_score(y_test, y_pred_test), "\n\n")
print("\n\nLearned Weight Vector: ", model1.weights, "\n\n")

print("\n\n---------------------------------Voted Perceptron---------------------------------\n")

model2 = model2.VotedPerceptron(learning_rate=0.01, n_iters=10)

model2.fit(traindf)

X = np.array(traindf.iloc[:,:-1])
y_pred_train = model2.predict(X)

#print(y_pred_train)


X = np.array(testdf.iloc[:,:-1])
y_pred_test = model2.predict(X)


y_train = traindf.iloc[:,-1:]

y_test = testdf.iloc[:,-1:]

#print(y_pred_test)
#print("\n\nTrain Accuracy: ", accuracy_score(y_train, y_pred_train), "\n\n")
w = np.array(model2.weights)
c = np.array(model2.C)
x, y = w.shape

for i in range(x):
    print("Distinct Weight Vector: ", w[i], "C: ", c[i], "\n")
print("\n\nTest Accuracy: ", accuracy_score(y_test, y_pred_test), "\n\n")
print("\n\nLearned Weight Vector: ", model2.weights[-1], "\n\n")


print("\n\n---------------------------------Averaged Perceptron---------------------------------\n")

model3 = model3.AveragedPerceptron(learning_rate=0.01, n_iters=10)

model3.fit(traindf)

#print(y_pred_train)

X = np.array(testdf.iloc[:,:-1])

y_pred_test = model2.predict(X)


y_train = traindf.iloc[:,-1:]

y_test = testdf.iloc[:,-1:]

print("\n\nLearned Weight Vector: ", model3.A, "\n\n")
print("\n\nTest Accuracy: ", accuracy_score(y_test, y_pred_test), "\n\n")




