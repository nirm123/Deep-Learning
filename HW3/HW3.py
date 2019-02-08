import numpy as np

import h5py
import time
import copy
from random import randint

# Function that calculates the softmax
def softmax(input_vector):
    exp_vec = np.exp(input_vector)
    return np.divide(exp_vec, np.sum(exp_vec))

# Function that calculates loss
def loss(theta, x, y):
    loss = 0
    for i in range(len(x)):
        soft = forward(theta, x[i])[0]
        loss -= np.log(soft[y[i]])
    return loss

# Function that applies reLU to a vector
def reLU(vector):
    vector[vector < 0] = 0
    return vector

# Function that calculates the derivative of the reLU function
def reLU_prime(vector):
    vector[vector > 0] = 1
    return vector

# Function that calcuate the gradient for all parameters
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
    dp_db = dp_dU

    '''
    dp_dC = np.matmul(dp_dU[:,None], output[2][:,None].T)
    sigma = np.matmul(theta[0][1].T, dp_dU)
    dp_db1 = np.multiply(sigma, reLU_prime(output[2])) 
    dp_dW = np.matmul(dp_db1[:,None], x_cur[:,None].T)
    return [dp_dW, dp_dC, dp_db1, dp_db2]
    '''
    return [dp_db]

# Function that calculates forward pass of neural network
def forward(weight, img):
    Z = np.empty([3,24,24])
    H = np.empty([3,24,24])
    tempI = 0
    tempJ = 0
    tempP = 0
    for p in range(3):
        for i in range(24):
            for j in range(24):
                Z[p][i][j] = np.sum(weight[0][p]*(img.reshape(28,28)[j:j+4,i:i+4]))
    
    H = np.copy(Z)
    reLU(H)
    
    U = np.empty(10)
    for q in range(10):
        for w in range(3):
            for i in range(24):
                for j in range(24):
                    U[q] += weight[1][q][p][i][j]*H[p][i][j]
    U += weight[2]
    soft_res = softmax(U)
    return_object = [soft_res, U, H, Z]
    return return_object

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
    W = np.random.uniform(-1,1,(10,3,24,24))
    b = np.random.uniform(-1,1,(10))
    theta = [K, W, b]

    alpha = 0.01
    iteration = 0
    epoch = 0
   
    # Training
    while True:

        # Calculate gradient for random element 
        grad = gradient(x_train, y_train, theta)
        theta[2] -= alpha * grad[0]    
        '''
        # Take step for all parameters based on alpha and gradient
        for i in range(4):
            theta[int(i/2)][i%2] = theta[int(i/2)][i%2] - alpha * grad[i]
        '''

        iteration += 1
        # Increment epoch every 20000 iterations
        if iteration == 1000:
            epoch += 1
            print(epoch)
            break
    # Testing
    test(x_test, y_test)
