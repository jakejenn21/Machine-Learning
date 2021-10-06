
import numpy as np
import random
 
def lms_gradient(x, y, w, train_size):
    sum = 0
    # print(x)
    # print(y)
    # print(w)
    # print(train_size)
    sum = (y-(np.dot(x,np.transpose(w))))
    res = -1 * sum
    return res

def sgd(x, y, n_vars=None, start=None, learn_rate=0.001, n_iter=100000, tolerance=1e-06, random_state=None):

    # Converting x and y to NumPy arrays
    x, y = np.array(x), np.array(y)
    n_obs = x.shape[0]
    if n_obs != y.shape[0]:
        raise ValueError("'x' and 'y' lengths do not match")

    # Initializing the values of the variables
    vector = np.zeros(x[0].shape)

    # Setting up and checking the learning rate
    learn_rate = np.array(learn_rate)
    if np.any(learn_rate <= 0):
        raise ValueError("'learn_rate' must be greater than zero")

    # Setting up and checking the maximal number of iterations
    n_iter = int(n_iter)
    if n_iter <= 0:
        raise ValueError("'n_iter' must be greater than zero")

      # Setting up and checking the tolerance
    tolerance = np.array(tolerance)
    if np.any(tolerance <= 0):
        raise ValueError("'tolerance' must be greater than zero")

    # Setting the difference to zero for the first iteration
    diff = np.zeros(x[0].shape)

    # Performing the gradient descent loop
    for _ in range(n_iter):
        random_index = random.randint(0, len(x)-1)

        # Recalculating the difference
        grad = np.array(lms_gradient(x[random_index], y[random_index], vector, x.shape[0]))
        diff =  diff - learn_rate * grad

        # Checking if the absolute difference is small enough
        if np.all(np.abs(diff) <= tolerance):
            break

        # Updating the values of the variables
        vector += diff

    return [vector, learn_rate] if vector.shape else [vector.item(), learn_rate]