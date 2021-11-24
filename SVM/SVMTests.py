import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PrimalStochasticSVM as model1
import DualSVM as model2

from sklearn.metrics import accuracy_score

#read concrete dataset
traindf = pd.read_csv("SVM/bank-note/train.csv", header=None)
traindf.insert(0,'', float(1.0))
#print(traindf)
testdf = pd.read_csv("SVM/bank-note/test.csv", header=None)
testdf.insert(0,'', float(1.0))
#print(testdf)

 #numpy arrays
X_train = traindf.iloc[:,:-1]

y_train = traindf.iloc[:,-1]

y_train = np.where(y_train==1, 1, -1)

X_test = testdf.iloc[:,:-1]

y_test = testdf.iloc[:,-1]

y_test = np.where(y_test==1, 1, -1)



#SVM in the primal domain with stochastic sub-gradient descent

#Set the maximum epochs T to 100.

#Use the curve of the objective function (along with the number of updates) to diagnosis the convergence.

#hyperparameters

C = [100/873,500/873,700/873]

gamma_0 = 0.5
alpha = 0.001

print("\n\n#-----PRIMAL STOCHASTIC SUB-GRADIENT SVM (learning_rate = gamma_0/(1 + (gamma_0/alpha)*t)-----#\n\n")

for c in C:
    w = model1.primal_stochastic_SVM(traindf, 0, gamma_0, alpha, c, T=100)
    print("C = ", c)
    print("\n")
    print("Final w:", w)
    preds_train = model1.predict(traindf, w)
    print("Train Accuracy: ", accuracy_score(y_train, preds_train))
    preds_test = model1.predict(testdf, w)
    print("Test Accuracy: ", accuracy_score(y_test, preds_test))
    print("\n")


print("\n\n#--------PRIMAL STOCHASTIC SUB-GRADIENT SVM (learning_rate = gamma_0/(1 + t))----------#\n\n")


for c in C:
    w = model1.primal_stochastic_SVM(traindf, 1, gamma_0, alpha, c, T=100)
    print("C = ", c)
    print("\n")
    print("Final w:", w)
    preds_train = model1.predict(traindf, w)
    print("Train Accuracy: ", accuracy_score(y_train, preds_train))
    preds_test = model1.predict(testdf, w)
    print("Test Accuracy: ", accuracy_score(y_test, preds_test))
    print("\n")


#read concrete dataset
traindf = pd.read_csv("SVM/bank-note/train.csv", header=None)
traindf.insert(0,'', float(1.0))
#print(traindf)
testdf = pd.read_csv("SVM/bank-note/test.csv", header=None)
testdf.insert(0,'', float(1.0))
#print(testdf)

 #numpy arrays
X_train = traindf.iloc[:,:-1]
X_train = X_train.to_numpy()

#print("DUAL X", X_train)


y_train = traindf.iloc[:,-1]

y_train = y_train.to_numpy()
y_train = np.where(y_train==1, 1, -1)

#print("DUAL Y", y_train)

X_test = testdf.iloc[:,:-1]
X_test = X_test.to_numpy()

y_test = testdf.iloc[:,-1]

y_test = y_test.to_numpy()
y_test = np.where(y_test==1, 1, -1)

#hyperparameters

C = [100/873,500/873,700/873]

print("#----------------------------DUAL SVM (linear) -----------------------------#\n\n")

for c in C:
    print("C = ", c)
    print("\n")

    model = model2.DualSVM(kernel='linear', C=c, gamma=0.01)

    print("TRAIN:\n")
    support_idx,a,w,b = model.fit(X_train, y_train)
    print("Final w:", w)
    print("Final b:", b)
    print("\n")

    print("TEST:\n")
    a,w,b = model.fit(X_test, y_test)
    print("Final w:", w)
    print("Final b:", b)
    print("\n")

print("#----------------------------DUAL SVM (gaussian) -----------------------------#\n\n")

gammas = [0.1,0.5,1.0,5.0,100.0]
C = [500/873]

a_point_one = []
a_point_five = []
a_one = []
a_five = []
a_hundred = []
for gamma in gammas:
    print("gamma = ", gamma)
    print("\n")
    for c in C:

        print("C = ", c)
        print("\n")
        model = model2.DualSVM(kernel='gaussian', C=c, gamma=gamma)
        print("TRAIN:\n")
        support_idx,a,w,b = model.fit(X_train, y_train)
        # print(a.shape)
        # print(a)

        if(c == 500/873):
            if(gamma == 0.1):
                sv_point_one = support_idx
            if(gamma == 0.5):
                sv_point_five = support_idx
            if(gamma == 1.0):
                sv_one = support_idx
            if(gamma == 5.0):
                sv_five = support_idx
            if(gamma == 100.0):
                sv_hundred = support_idx

        print("Final w:", w)
        print("Final b:", b)
        print("\n")

        print("TEST:\n")
        support_idx,a,w,b = model.fit(X_test, y_test)
        print("Final w:", w)
        print("Final b:", b)
        print("\n")



print("\nTESTING OVERLAP\n")

print("C=500/873\n")

res1 = np.intersect1d(sv_point_one, sv_point_five).shape[0]
print("gamma 0.1 and 0.5 : ", res1)
res2 = np.intersect1d(sv_point_five, sv_one).shape[0]
print("gamma 0.5 and 1.0 : ", res2)
res3 = np.intersect1d(sv_one, sv_five).shape[0]
print("gamma 1.0 and 5.0 : ", res3)
res4 = np.intersect1d(sv_five, sv_hundred).shape[0]
print("gamma 5.0 and 100.0 : ", res4)












