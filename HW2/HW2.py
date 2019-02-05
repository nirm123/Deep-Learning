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


if __name__ == "__main__":
    # Load MNIST data
    MNIST = h5py.File('MNISTdata.hdf5', 'r')
    x_train = np.float32(MNIST['x_train'][:])
    y_train = np.int32(np.array(MNIST['y_train'][:,0]))
    x_test = np.float32(MNIST['x_test'][:])
    y_test = np.int32(np.array(MNIST['y_test'][:,0]))

    MNIST.close()

    # Initialize variables
    weight = [np.random.randn(50,784)/np.sqrt(784), np.random.randn(10,50)/np.sqrt(784)]
    bias = [np.random.randn(50)/np.sqrt(784), np.random.randn(10)/np.sqrt(784)]
    theta = [weight, bias]
    
    alpha = 0.01
    iteration = 0
    epoch = 0 

    # Training
    while True:
        grad = gradient(x_train, y_train, theta)
        for i in range(4):
            theta[int(i/2)][i%2] = theta[int(i/2)][i%2] - alpha * grad[i]
        iteration += 1
        print(iteration)
        if iteration == 1000:
            iteration = 0
            epoch += 1
            break
    '''
    # Training
    while True:
        grad = gradient(x_train, y_train, theta)
        dp_dU = gradient(x_select, y_select, forward_output)
        
        dp_db2 = np.sum(dp_dU)/5
        
        dpdC = np.empty([5,10,50])
        sigma = np.empty([5,50,1])
        sigma_prime = np.empty([5,50])
        dpdb1 = np.empty([5,50,1])
        dpdW = np.empty([5,50,784])
        for i in range(5):
            dpdC[i] = np.matmul(dp_dU[i][np.newaxis].T,forward_output[i][2][np.newaxis])
            sigma[i] = np.matmul(theta[0][1].T, dp_dU[i][:,None])
            for j in range(50):
                if forward_output[i][2][j] > 0:
                    sigma_prime[i][j] = 1
                else:
                    sigma_prime[i][j] = 0
            dpdb1[i] = np.multiply(sigma[i],sigma_prime[i][:,None])
            dpdW[i] = np.multiply(dpdb1[i],theta[0][0])
        
        dp_dC = np.sum(dpdC)/5
        dp_db1 = np.sum(dpdb1)/5
        dp_dW = np.sum(dpdW)/5

        theta[0][0] = theta[0][0] - alpha*dp_dW
        theta[0][1] = theta[0][1] - alpha*dp_dC
        theta[1][0] = theta[1][0] - alpha*dp_db1
        theta[1][1] = theta[1][1] - alpha*dp_db2

        iteration += 5
        if iteration == 60000:
            print('Progress')
            iteration = 0
            epoch += 1
            if epoch == 5:
                break
    '''
    '''
    L_prev = loss(weight, x_train, y_train)
    print(str(L_prev) + " 0")

    # Training
    while True:
        grad = gradient(x_train, y_train, weight)
        weight = np.subtract(weight, alpha*grad)
        iteration += 1
        if iteration % 50 == 0:
            L = loss(weight, x_train, y_train)
            print(str(L) + " " + str(iteration))
        if iteration == 20000:
            break

    print("\nNumber of Iterations: " + str(iteration))
    print("Final Alpha: " + str(alpha) + "\n")
    '''

    # Testing
    total_correct = 0
    for n in range(len(x_test)):
        y = y_test[n]
        x = x_test[n][:]
        p = np.argmax(forward(theta,x)[0])
        if p == y:
            total_correct += 1
    print(total_correct/np.float(len(x_test)))
