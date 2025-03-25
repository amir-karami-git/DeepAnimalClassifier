import numpy as np
from lr_utils import load_dataset
import h5py

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
print(train_set_x_orig.shape, train_set_y.shape, test_set_x_orig.shape, test_set_y.shape)
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.
def sigmoid(z):
    a = 1/ (1+ np.exp(-z))
    return a
def RELU(z):
    a = np.maximum(0, z)
    return a
def tanh(z):
    a = np.tanh(z)
    return a
def initialize_zero(dim, nodes, parameter):
    if parameter == "w":
        w = np.zeros([dim, nodes])
        return w
    b = np.zeros([nodes, dim])
    return b
def initialize_random(dim, nodes):
    w = np.random.rand(dim,nodes) * 0.00001
    print(w.shape)
    return w
def HE_initialize(dim, nodes):
    w = np.random.rand(dim, nodes) * np.sqrt(10/dim)
    return w
def forward_propagation(w1, w2, b1, b2, X):
    A1 = tanh(np.dot(w1.T, X) + b1)
    #print(A1, "\n\n")
    A2 = sigmoid(np.dot(w2.T, A1) + b2)
    #print(A2)
    return A2, A1
def backward_propagation(A2, A1, Y, X, W2):
    m = Y.shape[1]
    dZ2 = A2 - Y
    dW2 = (1/m) * np.dot(A1, dZ2.T)  # Shape (4, 1)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True) 
    dZ1 = np.dot(W2, dZ2) * (1 - A1**2)
    dW1 = (1/m) * np.dot(X, dZ1.T)  # Shape (12288, 4)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)  # Shape (4, 1)
    return dW1, db1, dW2, db2
    
def calculate_cost(Y, A):
    m = Y.shape[1]
    epsilon = 1e-8  # Small constant to avoid log(0)
    cost = -1 / m * np.sum(Y * np.log(A + epsilon) + (1 - Y) * np.log(1 - A + epsilon))
    return cost
def perdiction(Z):
    for i in range(Z.shape[1]):
        if Z[0][i] > 0.5:
            Z[0][i] = 1
        else:
            Z[0][i] = 0
    return Z
def model(X, Y, learning_rate = 0.1):
    w1 = HE_initialize(X.shape[0], 4)
    b1 = initialize_zero(1,4,"b")
    w2 = HE_initialize(4,1)
    b2 = initialize_zero(1,1,"b")
    for i in range(14000):
        A2, A1 = forward_propagation(w1, w2, b1, b2, X)
        #print(A)
        dw1, dw2, db1, db2 = backward_propagation(A2, A1, Y, X, w2)
        #print("dw1 = ", dw1,"\ndw2 = ", dw2,"\ndb1 = ", db1,"\ndb2 = ", db2)
        w1 -= learning_rate * dw1
        b1 -= learning_rate * db1
        w2 -= learning_rate * dw2
        b2 -= learning_rate * db2
        if i % 1000 == 0:
            print(calculate_cost(Y, A2))
    count = 0
    A2, A1 = forward_propagation(w1, w2, b1, b2, X)
    for i in range(Y.shape[1]):    
        A2 = perdiction(A2)
        if A2[0][i] == Y[0][i]:
            count += 1
    print(count/Y.shape[1] * 100)
    return w1, w2, b1, b2
def test_deep(Y, X, w1, w2, b1, b2):
    count = 0
    Z, A1 = forward_propagation(w1, w2, b1, b2, X)
    for i in range(Y.shape[1]):
        Z = perdiction(Z)
        if Z[0][i] == Y[0][i]:
            count += 1
    print(count/Y.shape[1] * 100)


w1, w2, b1, b2 = model(train_set_x, train_set_y)
test_deep(test_set_y, test_set_x, w1, w2, b1, b2)