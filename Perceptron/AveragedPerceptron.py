import numpy as np
from sklearn.utils import shuffle

#https://www.cs.utah.edu/~zhe/teach/pdf/Perceptron.pdf 
#page 89
class AveragedPerceptron:

    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weight = None
        self.A = None

    def fit(self, traindf):
        # print(traindf)
        n_samples, n_features = traindf.shape
        # print(n_samples, n_features)
        # init parameters
        self.weight = np.zeros(n_features-1)
        self.A = np.zeros(n_features-1)

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
                check = y[idx]*np.dot(self.weight.T,x_i)
                # print("check:", check)

                if(check <= 0):
                    # Perceptron update rule
                    self.weight = self.weight + (self.lr *y[idx]*x_i)

                self.weight = self.weight
                self.A = self.A + self.weight

            


    def predict(self, X):
        preds = []
        for x in X:
            preds.append(np.sign(np.dot(self.A.T,x)))
        
        return preds