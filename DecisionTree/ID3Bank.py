
import numpy as np
import csv
import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ID3Classifier as classifier
import numbers

from numpy import log2 as log
from pprint import pprint

def accuracy_score(test, pred):
    return np.mean(pred == test)

def make_binary(attributes, df):
    df['age'] = np.where(df['age'] >= attributes['age'], "yes", "no")

    df['balance'] = np.where(df['balance'] >= attributes['balance'], "yes", "no")

    df['day'] = np.where(df['day'] >= attributes['day'], "yes", "no")

    df['duration'] = np.where(df['duration'] >= attributes['duration'], "yes", "no")

    df['campaign'] = np.where(df['campaign'] >= attributes['campaign'], "yes", "no")

    df['pdays'] = np.where(df['pdays'] >= attributes['pdays'], "yes", "no")

    df['previous'] = np.where(df['previous'] >= attributes['previous'], "yes", "no")


#read car dataset 

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


print("------------------------------BANK DATASET--------------------------------\n")


traindf = pd.read_csv("DecisionTree/bank/train.csv", header=None)
traindf.columns = list(attributes.keys()) + ["label"]
#print(traindf)
testdf = pd.read_csv("DecisionTree/bank/test.csv", header=None)
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

make_binary(attributes, traindf)
make_binary(attributes, testdf)

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

#--------------INFORMATION GAIN ----------------------

print("------------------------------INFORMATION GAIN (TREAT UNKOWN AS VALUE)--------------------------------\n")

# initialize and fit model to different depths for Information Gain for train and test datasets

#TRAIN
x = range(1,17)
y= []
for i in x:
    model = classifier.ID3Classifier(i,0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    y.append(score)


#TEST
x1 = range(1,17)
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
plt.title("INFORMATION GAIN SPLITTING (TREAT UNKOWN AS VALUE)")
plt.ylabel("Accuracy")
plt.xlabel("Depth")

print("INFORMATION GAIN SPLITTING (TREAT UNKOWN AS VALUE)")
for i in x:
    print("Depth:", i, "TRAIN:", y[i-1], "TEST:", y1[i-1])

print("\n")



#--------------MAJORITY ERROR-------------------------

print("------------------------------MAJORITY ERROR (TREAT UNKOWN AS VALUE)--------------------------------\n")

# initialize and fit model to different depths for Majority Error

#TRAIN
x = range(1,17)
y= []
for i in x:
    model = classifier.ID3Classifier(i,1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    y.append(score)


#TEST
x1 = range(1,17)
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
plt.title("MAJORITY ERROR SPLITTING (TREAT UNKOWN AS VALUE)")
plt.ylabel("Accuracy")
plt.xlabel("Depth")


print("MAJORITY ERROR SPLITTING (TREAT UNKOWN AS VALUE)")
for i in x:
    print("Depth:", i, "TRAIN:", y[i-1], "TEST:", y1[i-1])


print("\n")


# #--------------------GINI INDEX---------------------------------------

print("------------------------------GINI INDEX (TREAT UNKOWN AS VALUE)--------------------------------\n")


# # initialize and fit model to different depths for Gini Index

#TRAIN
x = range(1,17)
y= []
for i in x:
    model = classifier.ID3Classifier(i,2)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    y.append(score)


#TEST
x1 = range(1,17)
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
plt.title("GINI INDEX SPLITTING (TREAT UNKOWN AS VALUE)")
plt.ylabel("Accuracy")
plt.xlabel("Depth")

plt.show()

print("GINI INDEX SPLITTING (TREAT UNKOWN AS VALUE)")
for i in x:
    print("Depth:", i, "TRAIN:", y[i-1], "TEST:", y1[i-1])

print("\n")


#--------------INFORMATION GAIN UNKNOWN ----------------------

print("------------------------------INFORMATION GAIN (TREAT UNKNOWN AS MAJORITY)--------------------------------\n")


# initialize and fit model to different depths for Information Gain for train and test datasets

#TRAIN
x = range(1,17)
y= []
for i in x:
    model = classifier.ID3Classifier(i,0, True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    y.append(score)


#TEST
x1 = range(1,17)
y1= []
for i in x1:
    model = classifier.ID3Classifier(i,0, True)
    model.fit(X_test, y_test)
    y_pred = model.predict(X_train)
    score = accuracy_score(y_train, y_pred)
    y1.append(score)


plot4 =plt.figure(4)
plt.plot(x,y,'r-',label='Train')
plt.plot(x1,y1,'b-',label='Test')
plt.legend()
plt.title("Information Gain Splitting")
plt.ylabel("Accuracy")
plt.xlabel("Depth")

print("INFORMATION GAIN SPLITTING (TREAT UNKNOWN AS MAJORITY)")
for i in x:
    print("Depth:", i, "TRAIN:", y[i-1], "TEST:", y1[i-1])

print("\n")



#--------------MAJORITY ERROR-------------------------

print("------------------------------MAJORITY ERROR (TREAT UNKNOWN AS MAJORITY)--------------------------------\n")

# initialize and fit model to different depths for Majority Error

#TRAIN
x = range(1,17)
y= []
for i in x:
    model = classifier.ID3Classifier(i,1, True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    y.append(score)


#TEST
x1 = range(1,17)
y1= []
for i in x1:
    model = classifier.ID3Classifier(i,1, True)
    model.fit(X_test, y_test)
    y_pred = model.predict(X_train)
    score = accuracy_score(y_train, y_pred)
    y1.append(score)

plot5 =plt.figure(5)
plt.plot(x,y,'r-',label='Train')
plt.plot(x1,y1,'b-',label='Test')
plt.legend()
plt.title("MAJORITY ERROR SPLITTING (TREAT UNKNOWN AS MAJORITY)")
plt.ylabel("Accuracy")
plt.xlabel("Depth")


print("MAJORITY ERROR SPLITTING (TREAT UNKNOWN AS MAJORITY)")
for i in x:
    print("Depth:", i, "TRAIN:", y[i-1], "TEST:", y1[i-1])


print("\n")


# #--------------------GINI INDEX---------------------------------------
print("------------------------------GINI INDEX (TREAT UNKNOWN AS MAJORITY)--------------------------------\n")


# # initialize and fit model to different depths for Gini Index

#TRAIN
x = range(1,17)
y= []
for i in x:
    model = classifier.ID3Classifier(i,2, True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    y.append(score)


#TEST
x1 = range(1,17)
y1= []
for i in x1:
    model = classifier.ID3Classifier(i,2, True)
    model.fit(X_test, y_test)
    y_pred = model.predict(X_train)
    score = accuracy_score(y_train, y_pred)
    y1.append(score)

plot6 =plt.figure(6)
plt.plot(x,y,'r-',label='Train')
plt.plot(x1,y1,'b-',label='Test')
plt.legend()
plt.title("GINI INDEX SPLITTING (TREAT UNKNOWN AS MAJORITY)")
plt.ylabel("Accuracy")
plt.xlabel("Depth")

plt.show()

print("GINI INDEX SPLITTING (TREAT UNKNOWN AS MAJORITY)")
for i in x:
    print("Depth:", i, "TRAIN:", y[i-1], "TEST:", y1[i-1])

print("\n")

