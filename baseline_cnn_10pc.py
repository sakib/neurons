import numpy as np
from keras.models import Sequential  # Keras Model
from keras.layers import Dense, Dropout, Activation, Flatten  # Keras Layers
from keras.layers import Convolution2D, MaxPooling2D  # Keras CNN Layers
from keras.utils import np_utils
from matplotlib import pyplot as plt
from keras.datasets import mnist
from tempo_layer import DigitTempotronLayer
from dataset import MNIST

np.random.seed(7)  # for reproducibility

# With 50 training images and 50 testing images
# x_train should be 50*10*10 matrix, x_test should be 50*10*10 matrix
# y_train should be 50*1 vector, y_test should be 50*1 vector

dtl = DigitTempotronLayer()
dataset = MNIST(n_components=10)

samples = []
for digit in range(10): # 0->9
    for vector in dataset.sample(5, digit, digit): # 5 x 'digit'
        samples.append(vector)
samples = np.asarray(samples)

# Preprocess Input Matrices
X_train, X_test = [samples.astype('float32') for i in range(2)]
y_train, y_test = [dtl.get_layer_output()[1] for i in range(2)]

# Y_train should be 50*10 one hot matrix (encoded outputs)
# Y_test should be 50*10 one hot matrix (encoded outputs)
Y_train = np_utils.to_categorical(y_train)
Y_test = np_utils.to_categorical(y_test)
num_classes = Y_test.shape[1]

def keras_model():
    # create model
    model = Sequential()

    # first hidden layer with 20 neurons
    model.add(Dense(100, input_shape=(10,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # second hidden layer with 20 neurons
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # third hidden layer with 20 neurons
    model.add(Dense(100))
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

# training the model and saving metrics in history
history = model.fit(X_train, Y_train,
          batch_size=200, epochs=50, verbose=1,
          validation_data=(X_test, Y_test))


# plotting the metrics
fig = plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy (over 50 epochs)')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')

plt.tight_layout()
plt.show()


# Final evaluation of the model
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Baseline Model (10 PCgit status) Error: %.2f%%" % (100-scores[1]*100))
