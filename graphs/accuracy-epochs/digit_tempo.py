from tempo import Tempotron
from collections import defaultdict

RW = 'r' #'w' DON'T CHANGE THIS LINE PLEASE

# center pixel > avg of perimeter
def center_max(w):
	center = w[int(len(w)/2)][int(len(w)/2)]
	total = sum([sum(r) for r in w])
	n_perim = len(w)*len(w[0])-1.
	return center > (total-center)/n_perim

def pixel_avg(dataset):
	maxi = 300.
	mini = -300.
	for vector in dataset.data:
		if maxi < vector.max(): maxi = vector.max()
		if mini > vector.min(): mini = vector.min()
	def special_avg(w):
		total = sum([sum(r) for r in w])
		n_pixels = len(w)*len(w[0])
		return total/n_pixels > (maxi+mini)/2.
	return special_avg

# tempotron trained on single digit
class DigitTempotron(object):
	def __init__(self, dataset, digit, n_iters=10, n_neurons=64):
		self.digit = digit
		self.dataset = dataset # mnist object
		self.n_iters = n_iters # training
		self.tempotron = Tempotron(n_neurons)
		self.switch_fn = pixel_avg(self.dataset)
		self.train()

	def train(self, n_samples=5):
		with open('weights/{}.txt'.format(self.digit), 'r+') as f:
			if RW == 'w':
				samples = defaultdict()
				for digit in list(range(10)):
					samples[digit] = self.dataset.sample(n_samples, digit, self.digit)
				for x in range(self.n_iters):
					for digit, images in samples.items():
						for image in images:
							self.tempotron.train(image, self.switch_fn, digit == self.digit)
				for weight in self.tempotron.synapses:
					f.write(str(weight)+'\n')
			elif RW == 'r':
				wts = list(filter(lambda x: x != '', [line.strip() for line in f.readlines()]))
				self.tempotron.synapses = [float(wt) for wt in wts]


	def classify(self, image):
		return self.tempotron.classify(image, self.switch_fn)