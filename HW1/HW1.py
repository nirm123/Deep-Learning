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
        loss -= np.log(soft[y[i]])
    return loss

def gradient(x, y, theta):
    dtheta = np.copy(theta)
    for i in range(5):
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
        if i == 0:
            dtsum = dtheta
        else:
            dtsum += dtheta
    return  dtsum/5

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

    # Initialize variables
    weight = np.random.rand(10,784)
    alpha = 0.01
    iteration = 0
    L_prev = loss(weight, x_train, y_train)
    first = False
    print(str(L_prev) + " 0")

    # Training
    while True:
        grad = gradient(x_train, y_train, weight)
        weight = np.subtract(weight, alpha*grad)
        iteration += 1
        if iteration % 50 == 0:
            L = loss(weight, x_train, y_train)
            print(str(L) + " " + str(iteration))
            if L < 26000 and first == False:
                alpha /= 10
                first = True
            if L < 24000:
                break

    print("\nNumber of Iterations: " + str(iteration))
    print("Final Alpha: " + str(alpha) + "\n")
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
