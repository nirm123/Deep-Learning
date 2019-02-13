import numpy as np

import h5py
import time
import copy
from random import randint

# Function that calculates the softmax
def softmax(input_v):
    input_vector = input_v.copy()
    exp_vec = np.exp(input_vector)
    return np.divide(exp_vec, np.sum(exp_vec))

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

    num_channels = len(theta[0])
    kernel_dim = len(theta[0][0])
    convolution_dim = len(theta[1][0][0])
    
    sigma = np.zeros([num_channels,convolution_dim,convolution_dim])
    dK_dU = np.zeros([num_channels,kernel_dim,kernel_dim])
    dW_dU = np.zeros([num_channels,10,convolution_dim,convolution_dim])

    loss = -1 * np.log(output[0][y_cur])
    if np.argmax(output[0]) == y_cur:
        result = 1
    else:
        result = 0

    for i in range(num_channels):
        for j in range(10):
            sigma[i] += dp_dU[j] * theta[1][i][j]
            dW_dU[i][j] = dp_dU[j] * output[2][i]
    for channel in range(num_channels):
        dK_dU[channel] = convolution((reLU_prime(output[3][channel])*sigma[channel]),x_cur.reshape(28,28))
    return [dK_dU,dW_dU,db_dU,loss,result]

# Function that calculates forward pass of neural network
def forward(weight, img):
    num_channels = len(weight[0])
    kernel_dim = len(weight[0][0])
    convolution_dim = len(weight[1][0][0])
    img_dim = convolution_dim - 1 + kernel_dim

    Z = np.zeros([num_channels, convolution_dim, convolution_dim])
    H = np.zeros([num_channels, convolution_dim, convolution_dim])
    U = np.zeros([10])
    
    for channel in range(num_channels):
        Z[channel] = convolution(weight[0][channel],img.reshape(img_dim, img_dim))
    H = reLU(Z)
    for channel in range(num_channels):
        for k in range(10):
            for i in range(convolution_dim):
                for j in range(convolution_dim):
                    U[k] += weight[1][channel][k][i][j] * H[channel][i][j]
    soft_res = softmax(U)

    return [soft_res, U, H, Z]

def convolution(array1, array2):
    kernel_size = len(array1)
    dim = len(array2) - kernel_size + 1
    result = np.zeros([dim,dim])
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
    print('Test Accuracy: ' + str(total_correct/np.float(len(x_test))))

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
    num_channels = 10
    kernel_dim = 4
    convolution_dim = np.sqrt(len(x_train[0])) + 1 - kernel_dim
    
    K = np.random.randn(num_channels, kernel_dim, kernel_dim)/np.sqrt(784)
    W = np.random.randn(num_channels, 10, convolution_dim.astype(int), convolution_dim.astype(int))/np.sqrt(784)
    b = np.random.randn(10)/np.sqrt(784)
    theta = [K, W, b]

    alpha = 0.01
    iteration = 0
    
    epoch = 3
    iteration_epoch = 20000
    total_loss = 0
    total_accuracy = 0

    for i in range(epoch):
        if i != 0:
            alpha /= 10
        for j in range(iteration_epoch):
            grad = gradient(x_train, y_train, theta)
            for channel in range(3):
                theta[0][channel] -= alpha * grad[0][channel]
                theta[1][channel] -= alpha * grad[1][channel]
            theta[2] -= alpha * grad[2]
        
            total_loss += grad[3]
            total_accuracy += grad[4]

            if j % 1000 == 0 and j > 0:
                print(j + i * 20000)
                if j % 5000 == 0:
                    print('Average Loss: ' + str(total_loss/(5000)))
                    print('Average Accuracy: ' + str(total_accuracy/(5000)) + '\n')
                else:
                    print('Average Loss: ' + str(total_loss/(j%5000)))
                    print('Average Accuracy: ' + str(total_accuracy/(j%5000)) + '\n')
                if j % 5000 == 0:
                    total_loss = 0
                    total_accuracy = 0
                if j == 4000 and i == 0:
                    alpha /= 10

    # Testing
    test(x_test, y_test)
