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


#------------------------------CAR DATASET--------------------------------

#read car dataset 

traindf = pd.read_csv("DecisionTree/car/train.csv", header=None)
traindf.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']
#print(traindf)
testdf = pd.read_csv("DecisionTree/car/test.csv", header=None)
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


#--------------INFORMATION GAIN----------------------


# initialize and fit model to different depths for Information Gain for train and test datasets

#TRAIN
x = range(1,7)
y= []
for i in x:
    model = classifier.ID3Classifier(i,0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    y.append(score)

#TEST
x1 = range(1,7)
y1= []
for i in x1:
    model = classifier.ID3Classifier(i,0)
    model.fit(X_test, y_test)
    y_pred = model.predict(X_train)
    score = accuracy_score(y_train, y_pred)
    y1.append(score)

plot1 =plt.figure(1)
plt.plot(x,y,'r-',label='Train')
plt.plot(x1,y1,'b-',label='Test')
plt.legend()
plt.title("Information Gain Splitting")
plt.ylabel("Accuracy")
plt.xlabel("Depth")


print("INFORMATION GAIN SPLITTING")
for i in x:
    print("Depth:", i, "TRAIN:", y[i-1], "TEST:", y1[i-1])


print("\n")

#--------------MAJORITY ERROR-------------------------

# initialize and fit model to different depths for Majority Error

#TRAIN
x = range(1,7)
y= []
for i in x:
    model = classifier.ID3Classifier(i,1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    y.append(score)

#TEST
x1 = range(1,7)
y1= []
for i in x1:
    model = classifier.ID3Classifier(i,1)
    model.fit(X_test, y_test)
    y_pred = model.predict(X_train)
    score = accuracy_score(y_train, y_pred)
    y1.append(score)

plot2 =plt.figure(2)
plt.plot(x,y,'r-',label='Train')
plt.plot(x1,y1,'b-',label='Test')
plt.legend()
plt.title("Majority Error Splitting")
plt.ylabel("Accuracy")
plt.xlabel("Depth")


print("MAJORITY ERROR SPLITTING")
for i in x:
    print("Depth:", i, "TRAIN:", y[i-1], "TEST:", y1[i-1])


print("\n")


#--------------------GINI INDEX---------------------------------------


# initialize and fit model to different depths for Gini Index

#TRAIN
x = range(1,7)
y= []
for i in x:
    model = classifier.ID3Classifier(i,2)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    y.append(score)

#TEST
x1 = range(1,7)
y1= []
for i in x1:
    model = classifier.ID3Classifier(i,2)
    model.fit(X_test, y_test)
    y_pred = model.predict(X_train)
    score = accuracy_score(y_train, y_pred)
    y1.append(score)

plot3 =plt.figure(3)
plt.plot(x,y,'r-',label='Train')
plt.plot(x1,y1,'b-',label='Test')
plt.legend()
plt.title("Gini Index Splitting")
plt.ylabel("Accuracy")
plt.xlabel("Depth")


print("GINI INDEX SPLITTING")
for i in x:
    print("Depth:", i, "TRAIN:", y[i-1], "TEST:", y1[i-1])


print("\n")

plt.show()



