import numpy as np
from lr_utils import load_dataset
import h5py
import time

# implementing activation functions

# sigmoind function
def sigmoid(z):
    a = 1/ (1+ np.exp(-z))
    return a

# RELU function
def RELU(z):
    a = np.maximum(0, z)
    return a
# tanh function
def tanh(z):
    a = np.tanh(z)
    return a

# parameter initialization method

#zero method
def initialize_zero_W(size_of_w):
    w = np.zeros([size_of_w, 1])
    return w
def initialize_zero_b(size_of_b):
    b = np.zeros([size_of_b])
    
    return b

# random method
def initialize_random(size_of_w):
    w = np.random.rand(size_of_w,1)
    print(w.shape)
    return w

# he imethod
#def initialize_he():

def forward_propagation(W, X, b):
    z = sigmoid(np.dot(W.T, X) + b)
    return z

def backward_propagation(A, Y, X):
    m = Y.shape[1]
    dw = 1 / m * np.dot(X, (A - Y).T)
    db = 1 / m * np.sum(A - Y)
    return dw, db
def calculate_cost(Y, A):
    m = Y.shape[1]
    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    return cost

#prediction
def perdiction(Z):
    for i in range(Z.shape[1]):
        if Z[0][i] > 0.5:
            Z[0][i] = 1
        else:
            Z[0][i] = 0
    return Z
        
# model function
def model(X, Y, learning_rate):
    W = initialize_zero_W(X.shape[0])
    b = initialize_zero_b(1)
    for i in range(4000):
        A = forward_propagation(W, X, b)
        dw, db = backward_propagation(A, Y, X)
        W -= learning_rate * dw
        b -= learning_rate * db
        if i % 1000 == 0:
            print(calculate_cost(Y, A))
    count = 0
    # accuricy in train
    Z = forward_propagation(W, X, b)
    Z = perdiction(Z)
    for i in range(Y.shape[1]):    
        if Z[0][i] == Y[0][i]:
            count += 1
    print(count/Y.shape[1] * 100)
    return W, b
# accuricy in test
def test_deep(Y, X, W, b):
    count = 0
    Z = forward_propagation(W, X, b)
    Z = perdiction(Z)
    for i in range(Y.shape[1]):
        if Z[0][i] == Y[0][i]:
            count += 1
    print(count/Y.shape[1] * 100)

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
#print(train_set_x_orig.shape, train_set_y.shape, test_set_x_orig.shape, test_set_y.shape)
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.

W, b = model(train_set_x, train_set_y, 0.015)
test_deep(test_set_y, test_set_x, W, b)




