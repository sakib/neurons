import numpy as np
from math import sqrt
from random import randint
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_mldata

class MNIST(object):
	def __init__(self, n_components=784):
		mnist = fetch_mldata('MNIST original')
		X_train, self.labels = mnist.data/255., mnist.target
		pca = PCA(n_components=n_components)
		X_transformed = pca.fit_transform(X_train)
		sq_dims = lambda v: (int(sqrt(len(v))), int(sqrt(len(v))))
		self.data = [np.reshape(X, sq_dims(X)) for X in X_transformed]

	def sample(self, n_samples, label):
		samples = []
		while len(samples) != n_samples:
			idx = randint(0, len(self.data))
			if self.labels[idx] == label:
				samples.append(self.data[idx])
		return samples