
import torch
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from torch import nn


class NN(nn.Module):
    def __init__(self, hidden_width, depth, activation):
        super(NN, self).__init__()
        layers = []
        input_layer = nn.Linear(4, hidden_width)
        if activation == "relu":
            activation_input = nn.ReLU()
        else:
            activation_input = nn.Tanh()

        layers.append(input_layer)
        layers.append(activation_input)
        for i in range(depth-1):
            hidden_layer = nn.Linear(hidden_width, hidden_width)
            if activation == "relu":
                activation_hidden = nn.ReLU()
            else:
                activation_hidden = nn.Tanh()

            layers.append(hidden_layer)
            layers.append(activation_hidden)

        output_layer = nn.Linear(hidden_width, 1)
        layers.append(output_layer)

        self.model = nn.Sequential(*layers)

        def init_weights_relu(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight)

        def init_weights_tanh(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)

        if activation == 'relu':
            self.model.apply(init_weights_relu)
        else:
            self.model.apply(init_weights_tanh)

    def forward(self, X):
        input = np.float32(X)
        out = torch.from_numpy(input)
        out.requires_grad = True
        return self.model(out)


# load dataset
# read bank note dataset
traindf = pd.read_csv("Neural Networks/bank-note/train.csv")
# print(traindf)
testdf = pd.read_csv("Neural Networks/bank-note/test.csv")
# print(testdf)

# separate data into input and output features

# numpy arrays
X_train = traindf.iloc[:, :-1]
y_train = traindf.iloc[:, -1]
y_train = np.where(y_train == 1, 1, -1)

X_test = testdf.iloc[:, :-1]
y_test = testdf.iloc[:, -1]
y_test = np.where(y_test == 1, 1, -1)

criterion = torch.nn.MSELoss(reduction='sum')

depth = [3, 5, 9]
width = [5, 10, 25, 50, 100]

print("\nTensorFlow Test/Train errors based on Width/Depth\n\n")

print("ReLu:\n")
for i in width:
    for j in depth:
        NeuralNet = NN(i, j, "relu")
        optimizer = torch.optim.Adam(NeuralNet.parameters())
        for t in range(1000):

            optimizer.zero_grad()

            # Forward pass: compute predicted y by passing x to the model.
            y_train_pred_tensor = NeuralNet.forward(X_train.values)

            y_train_np = np.float32(y_train.copy())
            y_train_tensor = torch.from_numpy(y_train_np)
            y_train_tensor.requires_grad = True

            loss = criterion(y_train_pred_tensor.flatten(), y_train_tensor)

            # Backward pass: compute gradient of the loss with respect to model
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its parameters
            optimizer.step()

        train_pred = NeuralNet.forward(X_train.values)

        train_pred[train_pred >= 0.0] = 1.0

        train_pred[train_pred < 0.0] = -1.0

        train_pred = train_pred.detach().numpy().flatten()

        train_error = 1 - np.mean(train_pred == y_train)

        test_pred = NeuralNet.forward(X_test.values)

        test_pred[test_pred >= 0.0] = 1.0

        test_pred[test_pred < 0.0] = -1.0

        test_pred = test_pred.detach().numpy().flatten()

        test_error = 1 - np.mean(test_pred == y_test)

        print(f"width: {i}, depth:{j}, train error: {train_error.round(3)}, test error: {test_error.round(3)}")

print("\n")
print("Tanh:\n")
for i in width:
    for j in depth:
        NeuralNet = NN(i, j, "tanh")
        optimizer = torch.optim.Adam(NeuralNet.parameters())
        for t in range(1000):

            optimizer.zero_grad()

            # Forward pass: compute predicted y by passing x to the model.
            y_train_pred_tensor = NeuralNet.forward(X_train.values)

            y_train_np = np.float32(y_train.copy())
            y_train_tensor = torch.from_numpy(y_train_np)
            y_train_tensor.requires_grad = True

            loss = criterion(y_train_pred_tensor.flatten(), y_train_tensor)

            # Backward pass: compute gradient of the loss with respect to model
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its parameters
            optimizer.step()

        train_pred = NeuralNet.forward(X_train.values)

        train_pred[train_pred >= 0.0] = 1.0

        train_pred[train_pred < 0.0] = -1.0

        train_pred = train_pred.detach().numpy().flatten()

        train_error = 1 - np.mean(train_pred == y_train)

        test_pred = NeuralNet.forward(X_test.values)

        test_pred[test_pred >= 0.0] = 1.0

        test_pred[test_pred < 0.0] = -1.0

        test_pred = test_pred.detach().numpy().flatten()

        test_error = 1 - np.mean(test_pred == y_test)

        print(f"width: {i}, depth:{j}, train error: {train_error.round(3)}, test error: {test_error.round(3)}")

print("\n")
