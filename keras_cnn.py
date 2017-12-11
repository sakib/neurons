import numpy as np
from keras.models import Sequential  # Keras Model
from keras.layers import Dense, Dropout, Activation, Flatten  # Keras Layers
from keras.layers import Convolution2D, MaxPooling2D  # Keras CNN Layers
from keras.utils import np_utils
from matplotlib import pyplot as plt
from keras.datasets import mnist


np.random.seed(7)  # for reproducibility

# load data, output matrices from tempotron
# With 50 training images and 50 testing images
# X_train should be 50*10 matrix, X_test should be 50*10 matrix
# y_train should be 50*1 vector, y_test should be 50*1 vector
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess Input Matrices
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalize voltage inputs
max_voltage = 255 # need to change this number
X_train = X_train / max_voltage
X_test = X_test / max_voltage

# Y_train should be 50*10 one hot matrix (encoded outputs)
# Y_test should be 50*10 one hot matrix (encoded outputs)
Y_train = np_utils.to_categorical(y_train)
Y_test = np_utils.to_categorical(y_test)
num_classes = Y_test.shape[1]

def keras_model():
    # create model
    model = Sequential()

    # first hidden layer with 20 neurons
    model.add(Dense(20, input_shape=(10,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # second hidden layer with 20 neurons
    model.add(Dense(20))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # output layer with 10 neurons for 10 classes
    model.add(Dense(10))
    model.add(Activation('softmax'))

    # compiling the sequential model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    return model

# build the model
model = keras_model()
# Fit the model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Keras Model Error: %.2f%%" % (100-scores[1]*100))
