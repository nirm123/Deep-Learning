import numpy as np

import h5py
import time
import copy
from random import randint

def softmax(input_vector):
    exp_vec = np.exp(input_vector)
    return np.divide(exp_vec, np.sum(exp_vec))

def loss(theta, x, y):
    loss = 0
    for i in range(len(x)):
        soft = forward(theta, x[i])[0]
        loss -= np.log(soft[y[i]])
    return loss

def reLU(vector):
    return vector.clip(min=0)

def reLU_prime(vector):
    vector[vector > 0] = 1
    return vector

def gradient(x, y, theta):
    elem = np.random.randint(0, len(x))
    x_cur = x[elem]
    y_cur = y[elem]
    
    output = forward(theta, x_cur)

    dp_dU = np.empty(10)
    for z in range(len(dp_dU)):
            softZ = output[0][z]
            if z == y_cur:
                dp_dU[z] = -1*(1 - softZ)
            else:
                dp_dU[z] = -1*(-softZ)
    dp_db2 = dp_dU
    dp_dC = np.matmul(dp_dU[:,None], output[2][:,None].T)
    sigma = np.matmul(theta[0][1].T, dp_dU)
    dp_db1 = np.multiply(sigma, reLU_prime(output[2])) 
    dp_dW = np.matmul(dp_db1[:,None], x_cur[:,None].T)
    return [dp_dW, dp_dC, dp_db1, dp_db2]

def forward(weight, img):
    Z = np.add(np.matmul(weight[0][0], img), weight[1][0])
    H = reLU(Z)
    U = np.add(np.matmul(weight[0][1], H), weight[1][1])
    soft_res = softmax(U)
    return_object = [soft_res, Z, H, U]
    return return_object

def test(x_given, y_given):
    total_correct = 0
    for n in range(len(x_given)):
        y = y_given[n]
        x = x_given[n][:]
        p = np.argmax(forward(theta,x)[0])
        if p == y:
            total_correct += 1
    print(total_correct/np.float(len(x_test)))

if __name__ == "__main__":
    # Load MNIST data
    MNIST = h5py.File('MNISTdata.hdf5', 'r')
    x_train = np.float32(MNIST['x_train'][:])
    y_train = np.int32(np.array(MNIST['y_train'][:,0]))
    x_test = np.float32(MNIST['x_test'][:])
    y_test = np.int32(np.array(MNIST['y_test'][:,0]))

    MNIST.close()

    # Initialize variables
    weight = [np.random.randn(64,784)/np.sqrt(784), np.random.randn(10,64)/np.sqrt(784)]
    bias = [np.random.randn(64)/np.sqrt(784), np.random.randn(10)/np.sqrt(784)]
    theta = [weight, bias]
    
    alpha = 0.01
    iteration = 0

    # Training
    while True:
        grad = gradient(x_train, y_train, theta)
    
        for i in range(4):
            theta[int(i/2)][i%2] = theta[int(i/2)][i%2] - alpha * grad[i]
        
        iteration += 1
        
        if iteration == 60000:
            alpha = alpha/10
        if iteration == 120000:
            break
    
    # Testing
    test(x_test, y_test)
