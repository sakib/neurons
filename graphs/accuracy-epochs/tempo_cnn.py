import numpy as np
from keras.models import Sequential  # Keras Model
from keras.layers import Dense, Dropout, Activation, Flatten  # Keras Layers
from keras.layers import Convolution2D, MaxPooling2D  # Keras CNN Layers
from keras.utils import np_utils
from matplotlib import pyplot as plt
from keras.datasets import mnist
from tempo_layer import DigitTempotronLayer


np.random.seed(7)  # for reproducibility

# load data, output matrices from tempotron
# With 50 training images and 50 testing images
# x_train should be 50*10 matrix, x_test should be 50*10 matrix
# y_train should be 50*1 vector, y_test should be 50*1 vector
dtl = DigitTempotronLayer()
x_train = dtl.get_layer_output()[0]
x_test = x_train
y_train = dtl.get_layer_output()[1]
y_test = y_train
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#x_train = x_train.reshape(x_train.shape[0], 784)
#x_test = x_test.reshape(x_test.shape[0], 784)

# Preprocess Input Matrices
X_train = x_train.astype('float32')
X_test = x_test.astype('float32')

# normalize voltage inputs
max_voltage = 90
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
          batch_size=200, epochs=50,
          verbose=2,
          validation_data=(X_test, Y_test))

# plotting the metrics
fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')

plt.subplot(2,1,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')

plt.tight_layout()
plt.show()

scores = model.evaluate(X_test, Y_test, verbose=0)
print("Keras Model Error: %.2f%%" % (100-scores[1]*100))
