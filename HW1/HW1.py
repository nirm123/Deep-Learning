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
        soft = softmax(np.matmul(theta, x[i]))
        loss -= np.log(soft[y])
        
    return loss/len(x)

def gradient(x, y, theta):
    dtheta = np.copy(theta)
    elem = np.random.randint(0, len(x))
    x_cur = x[elem]
    y_cur = y[elem]
    for z in range(len(theta)):
        soft = softmax(np.matmul(theta, x_cur))
        softZ = soft[z]
        if z == y_cur:
            dtheta[z] = -1*(1 - softZ)*x_cur
        else:
            dtheta[z] = -1*(-softZ)*x_cur

    return dtheta

def predict(weight, img):
    soft_res = softmax(np.matmul(weight, img))
    return np.argmax(soft_res)
    

if __name__ == "__main__":
    # Load MNIST data
    MNIST = h5py.File('MNISTdata.hdf5', 'r')
    x_train = np.float32(MNIST['x_train'][:])
    y_train = np.int32(np.array(MNIST['y_train'][:,0]))
    x_test = np.float32(MNIST['x_test'][:])
    y_test = np.int32(np.array(MNIST['y_test'][:,0]))

    MNIST.close()

    # Initialize weights
    weight = np.random.rand(10,784)
    alpha = 0.01
    iteration = 0
    # Training
    while True:
        grad = gradient(x_train, y_train, weight)
        weight = np.subtract(weight, alpha*grad)
        #print(loss(weight, x_train, y_train))
        iteration += 1
        if iteration == 1000:
            break

    # Testing
    total_correct = 0
    for n in range(len(x_test)):
        y = y_test[n]
        x = x_test[n][:]
        p = predict(weight, x)
        # print(str(y) + " " + str(p))
        if p == y:
            total_correct += 1
    print(total_correct/np.float(len(x_test)))
