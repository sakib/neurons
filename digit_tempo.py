from tempo import Tempotron
from collections import defaultdict

# center pixel > avg of perimeter
def center_max(w):
	center = w[int(len(w)/2)][int(len(w)/2)]
	total = sum([sum(r) for r in w])
	n_perim = len(w)*len(w[0])-1.
	return center > (total-center)/n_perim

# tempotron trained on single digit
class DigitTempotron(object):
	def __init__(self, dataset, digit, n_iters=5, n_neurons=9):
		self.digit = digit
		self.dataset = dataset # mnist object
		self.n_iters = n_iters # training
		self.tempotron = Tempotron(n_neurons)
		self.train()
	
	def train(self, n_samples=5, switch_fn=center_max):
		samples = defaultdict()
		for digit in list(range(10)):
			samples[digit] = self.dataset.sample(n_samples if self.digit == digit else 1, digit)
		for x in range(self.n_iters):
			print('training tempotron on samples, iter {}...'.format(x))
			for digit, images in samples.items():
				for image in images:
					self.tempotron.train(image, switch_fn, digit == self.digit)
		#self.tempotron.print_synapses()

	def classify(self, image, switch_fn=center_max):
		return self.tempotron.classify(image, switch_fn)