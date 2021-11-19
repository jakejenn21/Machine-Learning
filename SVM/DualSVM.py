import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


class DualSVM:


    def __init__(self, kernel='linear', C=0, gamma=1, degree=3):

        if C is None:
            C=0
        if gamma is None:
            gamma = 1
        if kernel is None:
            kernel = 'linear'

        if kernel == 'linear':
            self.kernel_fn = self.linear_kernel
        if kernel == 'gaussian':
            self.kernel_fn = self.gaussian_kernel

        C = float(C)
        gamma = float(gamma)
        degree=int(degree)

        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.kernel = kernel
        self.tKt = None


    def gaussian_kernel(self, x, y):
        return np.exp(-1 * self.gamma*np.linalg.norm(x - y) ** 2 )

    def linear_kernel(self, x1, x2):
        return np.dot(x1, x2)

    
    def loss(self, a):
        """Evaluate the negative of the dual function at `a`.
        We access the optimization data (Gram matrix and target vector) from outer scope for convenience.
        :param a: dual variables
        """
        # at = a * t  # Hadamard product
        # return -(a.sum() - 0.5 * np.dot(at.T, np.dot(K, at))) 
        return -1 * (a.sum() - 0.5 * np.dot(a.T, np.dot(self.tKt, a)))

    
    def jac(self, a):
        """Calculate the Jacobian of the loss function (for the QP solver)"""
        return np.dot(a.T, self.tKt) - np.ones_like(a)


    def predict(self, test, X, t, k, a, b):
        """Form predictions on a test set.
        :param test: matrix of test data
        :param X: matrix of training data
        :param y: vector of training labels
        :param k: kernel used
        :param a: optimal dual variables (weights)
        :param b: optimal intercept
        """
        a_times_t = a * t
        y = np.empty(len(test))  # y is the array of predictions
        for i, s in enumerate(test):  # we skip kernel evaluation if a training data is not a support vector
            # evaluate the kernel between new data point and support vectors; 0 if not a support vector
            kernel_eval = np.array([k(s, x_m) if a_m > 0 else 0 for x_m, a_m in zip(X, a)])
            y[i] = a_times_t.dot(kernel_eval) + b
        return y


    def fit(self, X, y): 
        n_samples = X.shape[0]

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                # Kernel trick.
                if self.kernel == 'linear':
                    self.kernel_fn = self.linear_kernel
                    K[i, j] = self.linear_kernel(X[i], X[j])
                if self.kernel =='gaussian':
                    self.kernel_fn = self.gaussian_kernel
                    K[i, j] = self.gaussian_kernel(X[i], X[j])  

            
        self.tKt = y * K * y[:, np.newaxis]  # Hadamard product (quadratic form) 

        A = np.vstack((-np.eye(n_samples), np.eye(n_samples)))
        b = np.concatenate((np.zeros(n_samples), self.C * np.ones(n_samples)))


        constraints = ({'type': 'ineq', 'fun': lambda x: b - np.dot(A, x), 'jac': lambda x: -1 * A},
               {'type': 'eq', 'fun': lambda x: np.dot(x, y), 'jac': lambda x: y})


        # training
        a0 = np.random.rand(n_samples)  # initial guess
        res = minimize(self.loss, a0, jac=self.jac, constraints=constraints, method='SLSQP', options={})

        a = res.x  # optimal Lagrange multipliers
        a[np.isclose(a, 0)] = 0  # zero out nearly zeros
        a[np.isclose(a, self.C)] = self.C  # round the ones that are nearly C

        # points with a==0 do not contribute to prediction
        support_idx = np.where(0 < a)[0]  # index of points with a>0; i.e. support vectors
        margin_idx = np.where((0 < a) & (a < self.C))[0]  # index of support vectors that happen to lie on the margin, with 0<a<C
        print('Total %d data points, %d support vectors' % (n_samples, len(support_idx)))

        # find optimal weight vector
        optimal_w=0
        for n in support_idx:
            optimal_w += a[n]*y[n]*X[n]
        w = optimal_w/ len(support_idx)

        
        # still need to find the intercept term, b 
        a_times_y = a * y
        cum_b = 0
        for n in margin_idx:
            kernel_eval = np.array([self.kernel_fn(x_m, X[n]) if a_m > 0 else 0 for x_m, a_m in zip(X, a)])
            b = y[n] - a_times_y.dot(kernel_eval)
            cum_b += b
        b = cum_b / len(margin_idx)


        possibly_wrong_idx = np.where(a == self.C)[0]
        possibly_wrong_predictions = self.predict(X[possibly_wrong_idx], X, y, self.kernel_fn, a, b)
        num_wrong = np.sum((possibly_wrong_predictions * y[possibly_wrong_idx]) < 0)
        print('Classification accuracy: ' + str(1 - num_wrong / n_samples))

        return support_idx,a,w,b