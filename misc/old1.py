import sys, random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_mldata
from tempo import Tempotron
from math import sqrt

# sample x images of digit y from data
def sample(data, labels, n_samples, target):
	samples = []
	while len(samples) != n_samples:
		idx = random.randint(0, len(data))
		if labels[idx] == target:
			samples.append(data[idx])
	return samples

# center pixel > avg of perimeter
def center_max(w):
	center = w[int(len(w)/2)][int(len(w)/2)]
	total = sum([sum(r) for r in w])
	n_perim = len(w)*len(w[0])-1.
	return center > (total-center)/n_perim

n_neurons = 9
n_components = 25
n_train_iters = 10
training_digit = 0
false_digits = list(range(10))
false_digits.remove(training_digit)
mnist = fetch_mldata('MNIST original')
X_train, y_train = mnist.data/255., mnist.target
pca = PCA(n_components=n_components)
X_transformed = pca.fit_transform(X_train)
sq_dims = lambda v: (int(sqrt(len(v))), int(sqrt(len(v))))
data = [np.reshape(X, sq_dims(X)) for X in X_transformed]
tempotron = Tempotron(n_neurons)
switch_fns = {'center': center_max}

for name, switch_fn in switch_fns.items():
	valid_digits = sample(data, y_train, n_train_iters, training_digit)
	invalid_digits = [sample(data, y_train, 1, f) for f in false_digits]
	print('training...')
	for x in range(n_train_iters):
		for digit in valid_digits: tempotron.train(digit, switch_fn, True)
		for digit in invalid_digits: tempotron.train(digit[0], switch_fn, False)
	print('weights...')
	tempotron.print_synapses()
	print('classifying valid digits...')
	for digit in valid_digits:
		truth, t_max, v_max = tempotron.classify(digit, switch_fn)
		print('{}:\t{}\t{}\t{}'.format(training_digit, truth, t_max, v_max))
	i = 1
	for digit in invalid_digits:
		truth, t_max, v_max = tempotron.classify(digit[0], switch_fn)
		print('{}:\t{}\t{}\t{}'.format(i, truth, t_max, v_max)) # hack
		i += 1