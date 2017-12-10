from dataset import MNIST
from digit_tempo import DigitTempotron
#import matplotlib.pyplot as plt

n_samples = 3
training_digit = 0
dataset = MNIST(n_components=25)

print('training...')
tempotron = DigitTempotron(dataset, training_digit)
samples = {digit:dataset.sample(n_samples, digit) for digit in list(range(10))}

print('tempotron trained on {}...'.format(training_digit))
for digit, images in samples.items():
	for image in images:
		truth, t_max, v_max = tempotron.classify(image)
		print('{}:\t{}\t{}\t{}'.format(digit, truth, t_max, v_max))