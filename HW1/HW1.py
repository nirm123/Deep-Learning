import numpy as np

import h5py
import time
import copy
from random import randint

def softmax(input_vector):
    exp_vec = np.exp(input_vector)
    return np.divide(exp_vec, np.sum(exp_vec))

def loss(y_pred, y_train):
    return 0

def gradient():
    return 0

def predict(weight, bias, img):
    result = np.matmul(weight, img) + bias
    soft_res = softmax(result)
    return np.argmax(soft_res)
    

if __name__ == "__main__":
    # Load MNIST data
    MNIST = h5py.File('MNISTdata.hdf5', 'r')
    x_train = np.float32(MNIST['x_train'][:])
    y_train = np.int32(np.array(MNIST['y_train'][:,0]))
    x_test = np.float32(MNIST['x_test'][:])
    y_test = np.int32(np.array(MNIST['y_test'][:,0]))

    MNIST.close()

    # Initialize weights/bias
    weight = np.random.rand(10,784)
    bias = np.random.rand(10)

    # Training
    while True:
        y_pred = np.zeros(len(y_train))
        for i in range(len(x_train)):
            y_pred[i] = predict(weight, bias, x_train[i])

        if loss(y_pred, y_train) == 0:
            break

    # Testing
    total_correct = 0
    for n in range( len(x_test)):
        y = y_test[n]
        x = x_test[n][:]
        p = predict(weight, bias, x)
        print(str(y) + " " + str(p))
        if p == y:
            total_correct += 1

    print(total_correct/np.float(len(x_test)))
