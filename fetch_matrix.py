import numpy as np
from dataset import MNIST
from keras.datasets import mnist

samples = []
dataset = MNIST(n_components=100)
for digit in range(10): # 0->9
    for ten_by_ten_matrix in dataset.sample(5, digit, digit): # 5 x 'digit'
        samples.append(ten_by_ten_matrix)
samples = np.asarray(samples)
samples = samples.reshape(50, 100)
samples = samples.astype('float32')
print samples.shape
print samples[0]

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 784)
X_train = x_train.astype('float32')
print X_train.shape
print X_train[0]
