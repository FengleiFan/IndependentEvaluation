import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import lrp
import pandas as pd
from pylab import rcParams

import numpy as np
import tensorflow as tf
import h5py
import os

import matplotlib.pyplot as plt
import matplotlib.image as im
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches



rcParams['figure.figsize'] = 8, 10
mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)
Train_input = mnist.train.images
Train_label = mnist.train.labels

Test_input = mnist.test.images
Test_label = mnist.test.labels

batch_x = Test_input[0:10]
batch_y = Test_label[0:10]




csfont = {'fontname':'Times New Roman'}


plt.figure(dpi = 240)
gs1 = gridspec.GridSpec(5, 5)
gs1.update(wspace=0.025, hspace=0.09) # set the spacing between axes. 

ax=plt.subplot(gs1[0])
fig=plt.imshow(np.reshape(batch_x[0], (28,28)), origin="upper", cmap='gray')
plt.title('Digit',**csfont,fontsize=8)
plt.show()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
ax.set_xticklabels([])
ax.set_yticklabels([])

ax=plt.subplot(gs1[1])
fig=plt.imshow(np.reshape(im_list_simonyan[0], (28,28)), origin="upper", cmap='gray')
plt.title('Raw Gradient',**csfont,fontsize=8)
plt.show()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
ax.set_xticklabels([])
ax.set_yticklabels([])

ax=plt.subplot(gs1[2])
fig=plt.imshow(np.reshape(im_list_smilkov[0], (28,28)), origin="upper", cmap='gray')
plt.title('SmoothGrad',**csfont,fontsize=8)
plt.show()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
ax.set_xticklabels([])
ax.set_yticklabels([])


ax=plt.subplot(gs1[3])
fig=plt.imshow(np.reshape(im_list_Sundararaju[0], (28,28)), origin="upper", cmap='gray')
plt.title('IntegratedGrad',**csfont,fontsize=8)
plt.show()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
ax.set_xticklabels([])
ax.set_yticklabels([])


ax=plt.subplot(gs1[4])
fig=plt.imshow(np.reshape(im_list_DeepTaylor[0], (28,28)), origin="upper", cmap='gray')
plt.title('Deep Taylor',**csfont,fontsize=8)
plt.show()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
ax.set_xticklabels([])
ax.set_yticklabels([])


ax=plt.subplot(gs1[5])
fig=plt.imshow(np.reshape(batch_x[1], (28,28)), origin="upper", cmap='gray')

plt.show()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
ax.set_xticklabels([])
ax.set_yticklabels([])

ax=plt.subplot(gs1[6])
fig=plt.imshow(np.reshape(im_list_simonyan[1], (28,28)), origin="upper", cmap='gray')

plt.show()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
ax.set_xticklabels([])
ax.set_yticklabels([])

ax=plt.subplot(gs1[7])
fig=plt.imshow(np.reshape(im_list_smilkov[1], (28,28)), origin="upper", cmap='gray')

plt.show()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
ax.set_xticklabels([])
ax.set_yticklabels([])


ax=plt.subplot(gs1[8])
fig=plt.imshow(np.reshape(im_list_Sundararaju[1], (28,28)), origin="upper", cmap='gray')

plt.show()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
ax.set_xticklabels([])
ax.set_yticklabels([])


ax=plt.subplot(gs1[9])
fig=plt.imshow(np.reshape(im_list_DeepTaylor[1], (28,28)), origin="upper", cmap='gray')

plt.show()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
ax.set_xticklabels([])
ax.set_yticklabels([])

ax=plt.subplot(gs1[10])
fig=plt.imshow(np.reshape(batch_x[2], (28,28)), origin="upper", cmap='gray')

plt.show()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
ax.set_xticklabels([])
ax.set_yticklabels([])

ax=plt.subplot(gs1[11])
fig=plt.imshow(np.reshape(im_list_simonyan[2], (28,28)), origin="upper", cmap='gray')

plt.show()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
ax.set_xticklabels([])
ax.set_yticklabels([])

ax=plt.subplot(gs1[12])
fig=plt.imshow(np.reshape(im_list_smilkov[2], (28,28)), origin="upper", cmap='gray')

plt.show()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
ax.set_xticklabels([])
ax.set_yticklabels([])


ax=plt.subplot(gs1[13])
fig=plt.imshow(np.reshape(im_list_Sundararaju[2], (28,28)), origin="upper", cmap='gray')

plt.show()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
ax.set_xticklabels([])
ax.set_yticklabels([])


ax=plt.subplot(gs1[14])
fig=plt.imshow(np.reshape(im_list_DeepTaylor[2], (28,28)), origin="upper", cmap='gray')

plt.show()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
ax.set_xticklabels([])
ax.set_yticklabels([])

ax=plt.subplot(gs1[15])
fig=plt.imshow(np.reshape(batch_x[3], (28,28)), origin="upper", cmap='gray')

plt.show()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
ax.set_xticklabels([])
ax.set_yticklabels([])

ax=plt.subplot(gs1[16])
fig=plt.imshow(np.reshape(im_list_simonyan[3], (28,28)), origin="upper", cmap='gray')

plt.show()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
ax.set_xticklabels([])
ax.set_yticklabels([])

ax=plt.subplot(gs1[17])
fig=plt.imshow(np.reshape(im_list_smilkov[3], (28,28)), origin="upper", cmap='gray')

plt.show()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
ax.set_xticklabels([])
ax.set_yticklabels([])


ax=plt.subplot(gs1[18])
fig=plt.imshow(np.reshape(im_list_Sundararaju[3], (28,28)), origin="upper", cmap='gray')

plt.show()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
ax.set_xticklabels([])
ax.set_yticklabels([])


ax=plt.subplot(gs1[19])
fig=plt.imshow(np.reshape(im_list_DeepTaylor[3], (28,28)), origin="upper", cmap='gray')

plt.show()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
ax.set_xticklabels([])
ax.set_yticklabels([])


ax=plt.subplot(gs1[20])
fig=plt.imshow(np.reshape(batch_x[4], (28,28)), origin="upper", cmap='gray')

plt.show()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
ax.set_xticklabels([])
ax.set_yticklabels([])

ax=plt.subplot(gs1[21])
fig=plt.imshow(np.reshape(im_list_simonyan[4], (28,28)), origin="upper", cmap='gray')

plt.show()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
ax.set_xticklabels([])
ax.set_yticklabels([])

ax=plt.subplot(gs1[22])
fig=plt.imshow(np.reshape(im_list_smilkov[4], (28,28)), origin="upper", cmap='gray')

plt.show()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
ax.set_xticklabels([])
ax.set_yticklabels([])


ax=plt.subplot(gs1[23])
fig=plt.imshow(np.reshape(im_list_Sundararaju[4], (28,28)), origin="upper", cmap='gray')

plt.show()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
ax.set_xticklabels([])
ax.set_yticklabels([])


ax=plt.subplot(gs1[24])
fig=plt.imshow(np.reshape(im_list_DeepTaylor[4], (28,28)), origin="upper", cmap='gray')

plt.show()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
ax.set_xticklabels([])
ax.set_yticklabels([])