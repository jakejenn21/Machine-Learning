import numpy as np
import csv
import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import StochasticGradientDescent as sgd
import BatchGradientDescent as bgd

#read concrete dataset
traindf = pd.read_csv("Linear Regression/concrete/train.csv", header=None)
traindf.columns = ['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr','output']
#print(traindf)
testdf = pd.read_csv("Linear Regression/concrete/train.csv", header=None)
testdf.columns = ['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr','output']
#print(testdf)

# organize data into input and output
X_train = traindf.drop(columns="output")
y_train = traindf["output"]

X_test = testdf.drop(columns="output")
y_test = testdf["output"]

w = np.zeros(X_train.size)
train_size = X_train.size

weights= sgd.sgd(X_train, y_train)

print("Final Weight Vector SGD: ", weights[0])
print("Learning Rate: ", weights[1])

# weights, r = bgd.bgd(X_train, y_train)

# print("Final Weight Vector BGD: ", weights)
# print("Learning Rate: ", r)

