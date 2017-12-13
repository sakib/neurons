import numpy as np
from math import sqrt
from random import randint
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_mldata

RW = 'r' #'w'

class MNIST(object):
	def __init__(self, n_components=784, reshape=True):
		mnist = fetch_mldata('MNIST original')
		X_train, self.labels = mnist.data/255., mnist.target
		pca = PCA(n_components=n_components)
		X_transformed = pca.fit_transform(X_train)
		sq_dims = lambda v: (int(sqrt(len(v))), int(sqrt(len(v))))
		if reshape: self.data = [np.reshape(X, sq_dims(X)) for X in X_transformed]
		else: self.data = [X for X in X_transformed]

	def sample(self, n_samples, label, superlabel):
		samples = []
		with open('samples/{}.txt'.format(label), 'r+') as f:
			if RW == 'w':
				while len(samples) != n_samples:
					idx = randint(0, len(self.data))
					if self.labels[idx] == label:
						samples.append(self.data[idx])
						f.write(str(idx)+'\n')
			elif RW == 'r':
				idxs = list(filter(lambda x: x != '', [line.strip() for line in f.readlines()]))
				samples = [self.data[int(idx)] for idx in idxs]
		return samples[:(n_samples if label == superlabel else 1)]
