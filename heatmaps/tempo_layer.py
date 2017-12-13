import numpy as np
from dataset import MNIST
from digit_tempo import DigitTempotron

RW = 'r' #'w'

class DigitTempotronLayer(object):
	def __init__(self):
		# initialize dataset with PCA
		n_samples, tempotrons = 5, []
		dataset = MNIST(n_components=100)

		# training phase
		for training_digit in list(range(10)):
			print('training tempotron to recognize {}...'.format(training_digit))
			tempotrons.append(DigitTempotron(dataset, training_digit)) # auto-trains

		X_train, y_train = np.zeros((n_samples*10, 10)), np.zeros((n_samples*10, 1))

		# classification phase
		with open('X_train.txt', 'r+') as f:
			if RW == 'w':
				for i in range(n_samples*10):
					images = dataset.sample(n_samples, int(i/n_samples), int(i/n_samples)) # five sample vectors, each one of 0->9
					y_train[i] = int(i/n_samples)
					print('')
					for t in range(10):
						truth, t_max, X_train[i][t] = tempotrons[t].classify(images[i % n_samples])
						f.write(str(X_train[i][t])+'\n')
						print('tempotron {} classified digit {} v{} as:\t{}\t{}'\
							.format(t, int(i/n_samples), i%n_samples, truth, X_train[i][t]))
			if RW == 'r':
				vals = [float(y) for y in list(filter(lambda x: x != '', [line.strip() for line in f.readlines()]))]
				for i in range(len(vals)): # n_samples*10^2
					X_train[int(i/10)][i%10] = vals[i]
				for j in range(n_samples*10):
					y_train[j] = int(j/n_samples)

		self.tempotrons = tempotrons
		self.X_train, self.y_train = X_train, y_train
	
	def get_layer_output(self):
		return self.X_train, self.y_train


if __name__ == '__main__':
	dtl = DigitTempotronLayer()
	print(dtl.get_layer_output()[0]) # 50 x 10 matrix
