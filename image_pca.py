from PIL import Image
from numpy import *
from pylab import *
import PCA
from glob import glob

#from keras.datasets import mnist
# Get images from disk
"""
imageFolderPath = '/ilab/users/jl1806/cs443/Final_Proj/dataset/'
imagePath = glob(imageFolderPath + '/lena512.bmp')

img_arr = array([array(Image.open(img))
                  for img in imagePath],'f')
"""

(X_train, y_train), (X_test, y_test) = mnist.load_data()
img_arr = X_train[:5,:,:]

img = img_arr[0] # open one image to get size
m,n = img.shape[0:2] # get the size of the images
img_nbr = len(img_arr) # get the number of images

# create matrix to store all flattened images
img_matrix = array([img.flatten()
              for img in img_arr],'f')

# perform PCA
V,S,img_mean = PCA.pca(img_matrix)

# show some images (mean and 7 first modes)
for i in range(len(V)):
    img = Image.fromarray(V[i].reshape(m,n))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.save("{}.PNG".format(i))
img = Image.fromarray(img_mean.reshape(m,n))
if img.mode != 'RGB':
    img = img.convert('RGB')
img.save("mean.PNG")

