import numpy as np
from lr_utils import load_dataset
import h5py
import time
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
print(train_set_x_orig.shape, train_set_y.shape, test_set_x_orig.shape, test_set_y.shape)
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.
print(train_set_x.shape)
def sigmoid(z):
    a = 1/ (1+ np.exp(-z))
    return a
def RELU(z):
    a = np.maximum(0, z)
    return a
def tanh(z):
    a = np.tanh(z)
    return a
def initialize_zero(dimensions):
    layers = dimensions.shape[0]
    B = np.empty((layers, 1), dtype=object)

    for l in range(layers):
        b = np.zeros((dimensions[l][0], 1))
        B[l, 0] = b
    return B
def initialize_random(dimensions):
    layers = dimensions.shape[0]
    W = np.empty((layers, 1), dtype=object)
    for l in range(layers):
        w = np.random.rand(dimensions[l][1], dimensions[l][0])
        W[l,0] = w
    return W
def HE_initialize(dimensions):
    layers = dimensions.shape[0]
    W = np.empty((layers, 1), dtype=object)
    for l in range(layers):
        #print(dimensions[l][1])
        w = np.random.rand(dimensions[l][1], dimensions[l][0])* np.sqrt(2 /dimensions[l][1])
        W[l,0] = w
    return W
def forward_propagation(W, X, B):
    layers = W.shape[0]
    A = np.empty((layers+ 1, 1), dtype=object)
    A[0][0] = X
    for l in range(layers):
        z = np.dot(W[l][0].T, A[l][0]) + B[l][0]
        if l != layers-1:
            a = RELU(z)
        else:
            a = sigmoid(z)
        A[l+1,0] = a
    return A
def backward_propagation(W, B, A, Y):
    m = Y.shape[1]
    L = W.shape[0] 
    dW = np.empty_like(W)
    dB = np.empty_like(B)
    dZ = None

    A_final = A[L, 0]  # A[4]
    dZ = A_final - Y  # sigmoid derivative for BCE
    dW[L-1, 0] = np.dot(A[L-1, 0], dZ.T) / m  # A3 * dZ4.T
    dB[L-1, 0] = np.sum(dZ, axis=1, keepdims=True) / m

    for l in reversed(range(L - 1)):
        Z = np.dot(W[l, 0].T, A[l, 0]) + B[l, 0]  # recompute Z[l+1]
        dA = np.dot(W[l + 1, 0], dZ)  # backprop through W
        dZ = dA * (Z > 0)  # ReLU derivative

        dW[l, 0] = np.dot(A[l, 0], dZ.T) / m
        dB[l, 0] = np.sum(dZ, axis=1, keepdims=True) / m

    return dW, dB
def calculate_cost(Y, A):
    m = Y.shape[1]
    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    return cost
def perdiction(Z):
    for i in range(Z.shape[1]):
        if Z[0][i] > 0.5:
            Z[0][i] = 1
        else:
            Z[0][i] = 0
    return Z
def get_layer():
    Layers = int(input("input number of layers: "))
    neurons = []
    for i in range(Layers):
        neurons.append(int(input("input nerouns for " + str(i+1) + "th layer: ")))
    dim = []
    for i in range(Layers):
        if i == 0:
            dim.append([neurons[0], 12288])
        else:
            dim.append([neurons[i], neurons[i-1]])
        if i == (Layers - 1):
            dim.append([1, neurons[i]])
    dim = np.array(dim)
    return Layers, dim
def model(X, Y, learning_rate = 0.01):
    Layers, dim = get_layer()
    W = HE_initialize(dim)
    B = initialize_zero(dim)
    for i in range(5001):
        A = forward_propagation(W, X, B)
        dW, dB = backward_propagation(W, B, A, Y)
        W -= dW * learning_rate
        B -= dB * learning_rate
        if i%1000 ==0:
            print(calculate_cost(Y, A[-1][0]))
    A = forward_propagation(W, X, B)
    count = 0
    y = perdiction(A[-1][0])
    for i in range(Y.shape[1]):
        if y[0][i] == Y[0][i]:
            count += 1
    accuricy = count/ Y.shape[1] * 100
    print(accuricy)
    return W, B, accuricy
def test_deep(Y, X, W, B):
    count = 0
    A = forward_propagation(W, X, B)
    Z = perdiction(A[-1][0])
    for i in range(Y.shape[1]):
        if Z[0][i] == Y[0][i]:
            count += 1
    accuricy = count/Y.shape[1] * 100
    print(accuricy)
    return accuricy
W, B, acc_train = model(train_set_x, train_set_y)
acc_test = test_deep(test_set_y, test_set_x, W, B)