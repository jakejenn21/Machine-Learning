import numpy as np
import csv
import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import StochasticGradientDescent as sgd
import BatchGradientDescent as bgd

def cost(X, Y, w):
    sum = 0
    for i in range(1,X.shape[0]):
        sum += (Y[i]-w.T.dot(X[i]))**2
    return 1/2 * sum

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

model = sgd.sgd(X_train, y_train, learn_rate=0.0001, n_iter=1000, tolerance=1e-06, batch_size=1)

#Report

#
plt.figure(1)
plt.plot(model[0],model[1],'r-',label='Updates vs Cost')
plt.legend()
plt.title("Stochastic Gradient Descent")
plt.ylabel("Cost")
plt.xlabel("# of Updates")

print("\n")
#learned Weight vector
print("Learned Weight Vector: ", model[2])
print("\n")
#learning rate
print("Learning Rate: ", model[3])
print("\n")
#the cost function value of the test data with your learned weight vector
print("Cost Function Value: ", cost(np.array(X_test.values), np.array(y_test.values), np.array(model[2])))
print("\n")
print("\n")



#batch 

model = bgd.bgd(X_train, y_train, learn_rate=0.001, n_iter=1000, tolerance=1e-06, batch_size=10)

#Report

#
plt.figure(2)
plt.plot(model[0],model[1],'r-',label='Interations vs Cost')
plt.legend()
plt.title("Batch Gradient Descent")
plt.ylabel("Cost")
plt.xlabel("# of Iterations")

plt.show()

#learned Weight vector
print("Learned Weight Vector: ", model[2])
print("\n")
#learning rate
print("Learning Rate: ", model[3])
print("\n")
#the cost function value of the test data with your learned weight vector
print("Cost Function Value: ", cost(np.array(X_test.values), np.array(y_test.values), np.array(model[2])))

print("\n")
print("\n")

#solve analytically

X = np.array(X_train.values)
#print(X.shape)
Y = np.array(y_train.values)
#print(Y.shape)

lhs = np.linalg.inv(X.T.dot(X))
#print(lhs)
rhs = X.T.dot(Y)
#print(rhs)

w = lhs@rhs

print("Solved Analytically Final Weight :" , w)
print("\n")