import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_mldata

def show(matrix, s=''):
	plt.imshow(matrix, cmap=plt.cm.gray)
	plt.colorbar()
	plt.title(s)
	plt.show()

mnist = fetch_mldata('MNIST original')
data, labels = mnist.data, mnist.target

N = 5
C = 25
pca = PCA(n_components=C)
Z = pca.fit_transform(data)
Q = pca.transform([random.choice(data) for i in range(N)])
R = Q.dot(pca.components_)

print(Z.shape)
print(pca.components_.shape)
print(Q.shape)

for i in range(N):
	show(np.reshape(R[i], (28, 28)), 'pca with {} components'.format(C))

"""All the code necessary to do PCA compression:
mnist = fetch_mldata('MNIST original')
data, labels = mnist.data, mnist.target
pca = PCA(n_components=25)
Z = pca.fit_transform(data)

"""