import numpy as np
import random
 
def lms_gradient(x, y, w, train_size):
    sum = 0
    gradient = np.zeros(w.shape[0])
    for i in range(0, train_size-1):
        # print("y[i]:", y[i])
        # print("x[i]:",x[i])
        # print("w:",w)
        gradient.append(y[i]-(np.dot(x[i],np.transpose(w[i]))))

    return gradient

def cost(X, Y, w):
    sum = 0
    for i in range(1,X.shape[0]):
        sum += (Y[i]-w.T.dot(X[i]))**2
    return 1/2 * sum

def bgd(X, y, learn_rate=0.001, n_iter=100000, tolerance=1e-06, batch_size=1):
    # Converting x and y to NumPy arrays
    X, y = np.array(X), np.array(y)
    num_rows, num_cols = X.shape
    n_obs = X.shape[0]

    ts = []
    costs = []

    xy = np.c_[X.reshape(n_obs, -1), y.reshape(n_obs, 1)]
    # Initializing the random number generator
    rng = np.random.default_rng()

    w = np.zeros(num_cols)

    # Performing the gradient descent loop
    for t in range(n_iter):
        # Shuffle x and y
        rng.shuffle(xy)

        random_index = random.randint(1, num_rows)
        
        w_1 = w
            
        x_batch, y_batch = xy[0:random_index, :-1], xy[0:random_index, -1:]

        w_gradient = np.zeros(num_cols)

        for i in range(x_batch.shape[0]): # Calculating gradients for point in our K sized dataset
            prediction=np.dot(w.T,x_batch[i])
            w_gradient=w_gradient+(-2)*x_batch[i]*(y_batch[i]-(prediction))
        
        #Updating the weights(W) and Bias(b) with the above calculated Gradients
        w=w-learn_rate*(w_gradient/batch_size)
        
        ts.append(t)
        update_cost = cost(X, y, w)
        #print(update_cost)
        costs.append(update_cost)

        if(np.linalg.norm(w-w_1)< tolerance):
            break


    return [ts, costs, w, learn_rate] if w.shape else [ts, costs, w.item(), learn_rate]