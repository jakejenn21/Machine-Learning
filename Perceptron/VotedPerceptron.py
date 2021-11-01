import numpy as np
from sklearn.utils import shuffle

#https://www.cs.utah.edu/~zhe/teach/pdf/Perceptron.pdf 
#page 86
class VotedPerceptron:

    def __init__(self, learning_rate=0.01, n_iters=100):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = []
        self.C = []

    def fit(self, traindf):

        k = 0
        X = traindf.iloc[:,:-1]
        X = np.array(X)
        v = [np.ones_like(X)[0]]
        c = [0]

        for _ in range(self.n_iters):

            #Shuffle Dataset
            train_shuffle = shuffle(traindf)

            X = train_shuffle.iloc[:,:-1]
            X = np.array(X)

            y = train_shuffle.iloc[:,-1]
            y = np.array(y)
            y = np.where(y==1, 1, -1)


            # print("X:",X)
            # print("y:",y)

            for idx, x_i in enumerate(X):
                # print("y[idx] :", y[idx])
                # print("self.weights :", self.weights)
                # print("x_i :", x_i)
                pred = 1 if np.dot(v[k], x_i) > 0 else -1 # checks the sing of v*k
                if pred == y[idx]: # checks if the prediction matches the real Y
                    c[k] += 1 # increments c
                else:
                    v.append(np.add(v[k], np.dot(y[idx], x_i)))
                    c.append(1)
                    k += 1
                
            self.weights = v
            self.C = c


    def predict(self, X):
        preds = []
        for x in X:
            s = 0
            for w,c in zip(self.weights,self.C):
                # print("weights: ",w)
                # print("x: ", x)
                # print("c: ", c)
                # print("s: ", s)
                s = s + c*np.sign(np.dot(w,x))
            preds.append(np.sign(1 if s>= 0 else 0))
        return preds