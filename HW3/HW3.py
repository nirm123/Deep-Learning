import numpy as np

import h5py
import time
import copy
from random import randint

# Function that calculates the softmax
def softmax(input_v):
    input_vector = input_v.copy()
    print(input_vector)
    print()
    exp_vec = np.exp(input_vector)
    return np.divide(exp_vec, np.sum(exp_vec))

# Function that calculates loss
def loss(theta, x, y):
    loss = 0
    print("\nCalculating Loss:")
    for i in range(len(x)):
        soft = forward(theta, x[i])[0]
        loss -= np.log(soft[y[i]])
    return loss

# Function that applies reLU to a vector
def reLU(vec):
    vector = vec.copy()
    vector[vector < 0] = 0
    return vector

# Function that calculates the derivative of the reLU function
def reLU_prime(vec):
    vector = vec.copy()
    vector[vector > 0] = 1
    return vector

# Function that calcuate the gradient for all parameters
def gradient(x, y, theta):
    elem = np.random.randint(0, len(x))
    #elem = 30807
    print(elem)
    x_cur = x[elem]
    y_cur = y[elem]
    
    output = forward(theta, x_cur)

    dp_dU = np.zeros(10)
    for z in range(len(dp_dU)):
            softZ = output[0][z]
            if z == y_cur:
                dp_dU[z] = -1*(1 - softZ)
            else:
                dp_dU[z] = -1*(-softZ)
    db_dU = dp_dU

    sigma = np.zeros([3,25,25])
    dK_dU = np.zeros([3,4,4])
    dW_dU = np.zeros([3,10,25,25])

    for i in range(3):
        for j in range(10):
            sigma[i] += dp_dU[j] * theta[1][i][j]
            dW_dU[i][j] = dp_dU[j] * output[2][i]
    for channel in range(3):
        dK_dU[channel] = convolution((reLU_prime(output[3][channel])*sigma[channel]),x_cur.reshape(28,28))
    return [dK_dU,dW_dU,db_dU]

# Function that calculates forward pass of neural network
def forward(weight, img):
    Z = np.zeros([3,25,25])
    H = np.zeros([3,25,25])
    U = np.zeros([10])
    for channel in range(3):
        Z[channel] = convolution(weight[0][channel],img.reshape(28,28))
    H = reLU(Z)
    for channel in range(3):
        for k in range(10):
            for i in range(25):
                for j in range(25):
                    U[k] += weight[1][channel][k][i][j] * H[channel][i][j]
    soft_res = softmax(U)

    return [soft_res, U, H, Z]

def convolution(array1, array2):
    kernel_size = len(array1)
    dim = len(array2) - kernel_size + 1
    result = np.empty([dim,dim])
    for i in range(dim):
        for j in range(dim):
            result[i][j] = np.sum(array1*array2[i:i+kernel_size,j:j+kernel_size])
    return result


# Function that calculates accuracy of prediction
def test(x_given, y_given):
    total_correct = 0
    count = 0
    for n in range(len(x_given)):
        count += 1
        y = y_given[n]
        x = x_given[n][:]
        p = np.argmax(forward(theta,x)[0])
        if p == y:
            total_correct += 1
    print(total_correct/np.float(len(x_test)))

# Main
if __name__ == "__main__":
    # Load MNIST data
    MNIST = h5py.File('MNISTdata.hdf5', 'r')
    x_train = np.float32(MNIST['x_train'][:])
    y_train = np.int32(np.array(MNIST['y_train'][:,0]))
    x_test = np.float32(MNIST['x_test'][:])
    y_test = np.int32(np.array(MNIST['y_test'][:,0]))

    MNIST.close()

    # Initialize variables
    K = np.random.uniform(-1,1,(3,4,4))
    W = np.random.uniform(-1,1,(3,10,25,25))
    b = np.random.uniform(-1,1,(10))
    theta = [K, W, b]

    alpha = 0.01
    iteration = 0
    epoch = 0
    
    while True:
        grad = gradient(x_train, y_train, theta)
        for channel in range(3):
            theta[0][channel] -= alpha * grad[0][channel]
            theta[1][channel] -= alpha * grad[1][channel]
        theta[2] -= alpha * grad[2]

        print(theta[0])
        print(theta[1])
        print(theta[2])
        iteration += 1
        if iteration % 100 == 0:
            print(iteration)
        if iteration == 1000:
            break
    # Testing
    #test(x_test, y_test)
