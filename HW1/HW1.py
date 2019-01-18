import numpy as np

import h5py
import time
import copy
from random import randint

# Load MNIST data
MNIST = h5py.File('MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST['x_train'][:])
y_train = np.int32(np.array(MNIST['y_train'][:,0]))
x_test = np.float32(MNIST['x_test'][:])
y_test = np.int32(np.array(MNIST['y_test'][:,0]))

MNIST.close()

