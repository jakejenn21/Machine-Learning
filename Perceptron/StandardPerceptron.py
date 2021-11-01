import numpy as np
from sklearn.utils import shuffle


class StandardPerceptron:

    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, traindf):
        # print(traindf)
        n_samples, n_features = traindf.shape
        # print(n_samples, n_features)
        # init parameters
        self.weights = np.zeros(n_features-1)

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
                check = y[idx]*np.dot(self.weights.T,x_i)
                # print("check:", check)

                if(check <= 0):
                    # Perceptron update rule
                    update = self.weights + (self.lr *y[idx]*x_i)
                else:
                    continue

                self.weights = update

    def predict(self, X):
        linear_output = X@self.weights
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def _unit_step_func(self, x):
        return np.where(x>=0, 1, 0)