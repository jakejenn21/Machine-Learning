import numpy as np
 
def lms_gradient(x, y, w, train_size):
    sum = 0
    # print(x)
    # print(y)
    # print(w)
    # print(train_size)
    gradient = []
    for i in range(0, train_size):
        sum += (y[0]-(np.dot(x[i],np.transpose(w))))

    res = -1 * sum
    return res

def bgd(x, y, n_vars=None, start=None, learn_rate=0.1,
    decay_rate=0.0, batch_size=5, n_iter=10000, tolerance=1e-06, random_state=None
):
    # Converting x and y to NumPy arrays
    x, y = np.array(x), np.array(y)
    n_obs = x.shape[0]
    if n_obs != y.shape[0]:
        raise ValueError("'x' and 'y' lengths do not match")
    xy = np.c_[x.reshape(n_obs, -1), y.reshape(n_obs, 1)]
    # Initializing the random number generator
    seed = None if random_state is None else int(random_state)
    rng = np.random.default_rng(seed=seed)

    # Initializing the values of the variables
    vector = np.zeros(x[0].shape)

    # Setting up and checking the learning rate
    learn_rate = np.array(learn_rate)
    if np.any(learn_rate <= 0):
        raise ValueError("'learn_rate' must be greater than zero")

    # Setting up and checking the decay rate
    decay_rate = np.array(decay_rate)
    if np.any(decay_rate < 0) or np.any(decay_rate > 1):
        raise ValueError("'decay_rate' must be between zero and one")

    # Setting up and checking the size of minibatches
    batch_size = int(batch_size)
    if not 0 < batch_size <= n_obs:
                raise ValueError(
            "'batch_size' must be greater than zero and less than "
            "or equal to the number of observations"
        )

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
        # Shuffle x and y
        rng.shuffle(xy)

        # Performing minibatch moves
        for start in range(0, n_obs, batch_size):
            stop = start + batch_size
            x_batch, y_batch = xy[start:stop, :-1], xy[start:stop, -1:]

            w_prev = diff

            # Recalculating the difference
            grad = np.array(lms_gradient(x_batch, y_batch, vector, x_batch.size))
            diff = decay_rate * diff - learn_rate * grad

            # Checking if the absolute difference is small enough
            if np.all(np.abs(diff-w_prev) <= tolerance):
                break

            # Updating the values of the variables
            vector += diff

    return vector if vector.shape else vector.item()