import numpy as np
import matplotlib as mpl
from keras.models import Sequential  # Keras Model
from keras.layers import Dense, Dropout, Activation, Flatten  # Keras Layers
from keras.layers import Convolution2D, MaxPooling2D  # Keras CNN Layers
from keras.utils import np_utils
from matplotlib import pyplot as plt
from keras.datasets import mnist
from tempo_layer import DigitTempotronLayer
from dataset import MNIST

# With 50 training images and 50 testing images
# x_train should be 50*10*10 matrix, x_test should be 50*10*10 matrix
# y_train should be 50*1 vector, y_test should be 50*1 vector
dtl = DigitTempotronLayer()
dataset = MNIST(n_components=10, reshape=False)
np.random.seed(7)  # for reproducibility

# Training data
train_samples = []
for digit in range(10): # 0->9
    for ten_by_ten_matrix in dataset.sample(5, digit, digit): # 5 x 'digit'
        train_samples.append(ten_by_ten_matrix)
train_samples = np.asarray(train_samples).reshape(50, 10)
X_train = train_samples.astype('float32')
y_train = dtl.get_layer_output()[1]
Y_train = np_utils.to_categorical(y_train) # 50*10 one hot matrix (encoded outputs)

# Testing data
i = 0
test_samples = []
y_test = np.zeros((50, 1))
for digit in range(10): # 0->9
    for ten_by_ten_matrix in dataset.new_sample(5, digit):
        test_samples.append(ten_by_ten_matrix)
        y_test[i] = digit
        i += 1
test_samples = np.asarray(test_samples).reshape(50, 10)
X_test = test_samples.astype('float32')
Y_test = np_utils.to_categorical(y_test) # 50*10 one hot matrix (encoded outputs)
num_classes = Y_test.shape[1]


def keras_model(n_hidden_layers, n_neurons):
    # create model
    model = Sequential()
    # first hidden layer with 20 neurons
    model.add(Dense(100, input_shape=(10,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    for i in range(n_hidden_layers): # i'th hidden layer
        model.add(Dense(n_neurons))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
    # output layer with 10 neurons for 10 classes
    model.add(Dense(10))
    model.add(Activation('softmax'))
    # compiling the sequential model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    return model

a = np.zeros((5, 10))

for batch_size in [5, 50]:
    for n_hidden_layers in range(1, 6): # 1->5
        for n_neurons in range(10, 110, 10): # 10->100
            # build the model
            model = keras_model(n_hidden_layers, n_neurons)
            # training the model and saving metrics in history
            history = model.fit(X_train, Y_train,
                      batch_size=batch_size, epochs=50, verbose=0,
                      validation_data=(X_test, Y_test))

            a[n_hidden_layers-1][int(n_neurons/10)-1] = model.evaluate(X_test, Y_test, verbose=0)[1]
            print('bs: {}\th: {}\tn: {}\tacc: {}'.format(batch_size, n_hidden_layers,
                n_neurons, a[n_hidden_layers-1][int(n_neurons/10)-1]))

    plt.imshow(a, cmap=mpl.cm.get_cmap('Reds'), extent=[-0.5, 9.5, 0.5, 5.5])
    plt.clim(vmin=0., vmax=1.)
    plt.title('Accuracy of Base CNN, 10 PCs, batch {}'.format(batch_size))
    plt.ylabel('n_hidden_layers')
    plt.yticks(np.arange(1, 6, 1))
    plt.xlabel('n_neurons')
    plt.xticks(np.arange(10), (10*(x+1) for x in range(10)))
    plt.colorbar()
    plt.show()
