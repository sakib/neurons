from tempo import Tempotron
from keras.datasets import mnist

class Tempo_Classifier:
    """
    Attributes:
        x_train, x_test: uint8 array of grayscale image data with shape (num_samples, 28, 28).
        y_train, y_test: uint8 array of digit labels (integers in range 0-9) with shape (num_samples,).
    """

    def __init__(self, num_classes, tempo_windows):
        """
        Parameters:
            num_inputs: number of classifications
            tempo_windows: number of windows per tempotron; used in tempo init
        """
        #list of tempotrons, each of which takes an input set
        inputs = [Tempotron(tempo_windows) for i in range(num_classes)]

        #get mnist data
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()


if __name__ == '__main__':

    tc = Tempo_Classifier(5, 9)
