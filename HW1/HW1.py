import numpy as np

import h5py
import time
import copy
from random import randint


def softmax(input_vector):
    exp_vec = np.exp(input_vector)
    return np.divide(exp_vec, np.sum(exp_vec))

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
        y_predicted = np.copy(y_train)
        index = 0
        for img in x_train:
            result = np.matmul(weight, img) + bias
            soft_res = softmax(result)
            y_predicted[index] = np.argmax(soft_res)
            index += 1
        break
    print(y_predicted)
    print(y_train)
