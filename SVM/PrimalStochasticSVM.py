# We will  rst implement SVM in the primal domain with stochastic sub-gradient descent. 
# We will reuse the dataset for Perceptron implementation, namely: \bank-note.zip" in Canvas. 
# The training data are stored in the  le \classification/train.csv", consisting of 872 examples. 
# The test data are stored in \classification/test.csv", and comprise of 500 examples. 
# Set the maximum epochs T to 100. Don't forget to shuffle the training examples at the start of each epoch. 
# Use the curve of the objective function (along with the number of updates) to diagnosis the convergence. 
# Try the hyperparameter given.
# Don't forget to convert the labels to be in (-1,1)

# Use the schedule of learning rate: 
# gamma_t = gamma_0/(1+gamma_0/alpha*t)

# Please tune gamma_0 and alpha to ensure convergence. 
# For each setting of C, report your training and test error.

#Use the schedule:
# gamma_t = gamma_0/(1+gamma_0/alpha*t)

#Report the training and test error for each setting of C.

# For each C:
# report the differences between the model parameters learned from the two learning rate schedules
# as well as the differences between the training/test errors. 
# What can you conclude?