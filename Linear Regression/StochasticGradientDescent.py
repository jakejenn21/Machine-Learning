
import numpy as np
import random
import matplotlib.pyplot as plt
 
def stocastic_gradient(x, y, w, train_size):
    return (y-(x.dot(w.T))*x)

def cost(X, Y, w):
    sum = 0
    for i in range(1,X.shape[0]):
        sum += (Y[i]-w.T.dot(X[i]))**2
    return 1/2 * sum

def sgd(X, y, learn_rate=0.001, n_iter=100000, tolerance=1e-06, batch_size=1):
    # Converting x and y to NumPy arrays
    X, y = np.array(X), np.array(y)
    num_rows, num_cols = X.shape
    n_obs = X.shape[0]

    updates = []
    costs = []

    xy = np.c_[X.reshape(n_obs, -1), y.reshape(n_obs, 1)]
    # Initializing the random number generator
    rng = np.random.default_rng()

    w = np.zeros(num_cols)

    update_count = 0 
    
    # Performing the gradient descent loop
    for t in range(n_iter):

        # Shuffle x and y
        rng.shuffle(xy)

        # Performing minibatch moves
        for start in range(0, n_obs, batch_size):
            stop = start + batch_size
            x_batch, y_batch = xy[start:stop, :-1], xy[start:stop, -1:]

            w_gradient = np.zeros(num_cols)

            for i in range(batch_size): # Calculating gradients for point in our K sized dataset
                prediction=np.dot(w.T,x_batch[i])
                w_gradient=w_gradient+(-2)*x_batch[i]*(y_batch[i]-(prediction))
        
            #Updating the weights(W) and Bias(b) with the above calculated Gradients
            w=w-learn_rate*(w_gradient/batch_size)
            update_count+=1
        
        updates.append(update_count)
        update_cost = cost(X, y, w)
        #print(update_cost)
        costs.append(update_cost)


    return [updates, costs, w, learn_rate] if w.shape else [updates, costs, w.item(), learn_rate]

def predict(X,w):
    y_pred = []
    num_vars = X[0].shape
    for i in range(len(X)):
        temp = X
        X_test = temp.iloc[:,0:num_vars].values
        y = np.asscalar(np.dot(w,X_test[i]))
        y_pred.append(y)

    return np.array(y_pred)