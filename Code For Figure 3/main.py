from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import abc
import sys

import numpy as np
import pandas as pd
from sklearn import linear_model, preprocessing, cluster

import scipy.linalg as slin
import scipy.sparse.linalg as sparselin
import scipy.sparse as sparse
from scipy.optimize import fmin_ncg

import os.path
import time
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.keras import backend as k
from tensorflow.contrib.learn.python.learn.datasets import base

from influence.hessians import hessian_vector_product
from influence.dataset import DataSet




import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import pandas as pd
from pylab import rcParams

rcParams['figure.figsize'] = 8, 10

########### Data Loading

mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)


Train_input = mnist.train.images
Train_label = mnist.train.labels

Test_input = mnist.test.images
Test_label = mnist.test.labels


def get_influence_on_test_loss(sess, grad_total_loss_op, test_indices, train_idx=None, 
        approx_type='lissa', approx_params=None, force_refresh=True, test_description=None,
        X_train = Train_input, Y_train = Train_label, X_test = Test_input, Y_test = Test_label):
        # If train_idx is None then use X and Y (phantom points)
        # Need to make sure test_idx stays consistent between models
        # because mini-batching permutes dataset order



        test_grad_loss_no_reg_val = get_test_grad_loss_no_reg_val(sess, grad_loss_no_reg_op, X_test, Y_test, test_indices,batch_size=100 )

        print('Norm of test gradient: %s' % np.linalg.norm(test_grad_loss_no_reg_val[0]))

        start_time = time.time()

        if test_description is None:
            test_description = test_indices


        
        inverse_hvp = get_inverse_hvp_lissa(test_grad_loss_no_reg_val, sess, v_placeholder, hessian_vector,
                              batch_size=None,
                              scale=10, damping=0.0, num_samples=1, recursion_depth=1000)

        duration = time.time() - start_time
        print('Inverse HVP took %s sec' % duration)



        start_time = time.time()
        

        
        num_to_remove = 100
        predicted_loss_diffs = np.zeros([num_to_remove])            
        for counter in np.arange(num_to_remove):
                print(counter) 
                single_train_feed_dict = {x: X_train[counter, :], y_ : [Y_train[counter,:]]}      
                train_grad_loss_val = sess.run(grad_total_loss_op, feed_dict=single_train_feed_dict)
                predicted_loss_diffs[counter] = np.dot(np.concatenate(inverse_hvp), np.concatenate(train_grad_loss_val)) / num_to_remove          

           
        duration = time.time() - start_time
        print('Multiplying by %s train examples took %s sec' % (num_to_remove, duration))

        return predicted_loss_diffs


def get_test_grad_loss_no_reg_val(sess, grad_loss_no_reg_op, Test_input, Test_label, test_indices, batch_size=100):



        if test_indices is not None:
            num_iter = int(np.ceil(len(test_indices) / batch_size))

            test_grad_loss_no_reg_val = None
            for i in range(num_iter):
                start = i * batch_size
                end = int(min((i+1) * batch_size, len(test_indices)))

                test_feed_dict = fill_feed_dict_with_some_ex(x, y_, Test_input, Test_label, test_indices[start:end])

                temp = sess.run(grad_loss_no_reg_op, feed_dict=test_feed_dict)

                if test_grad_loss_no_reg_val is None:
                    test_grad_loss_no_reg_val = [a * (end-start) for a in temp]
                else:
                    test_grad_loss_no_reg_val = [a + b * (end-start) for (a, b) in zip(test_grad_loss_no_reg_val, temp)]

            test_grad_loss_no_reg_val = [a/len(test_indices) for a in test_grad_loss_no_reg_val]

        
        return test_grad_loss_no_reg_val    
    

    
def fill_feed_dict_with_all_but_one_ex(x, y_, data_images, data_labels, idx_to_remove):
        num_examples = data_images.shape[0]
        idx = np.array([True] * num_examples, dtype=bool)
        idx[idx_to_remove] = False
        feed_dict = {
            x: data_images[idx, :],
            y: data_labels[idx, :]
        }
        return feed_dict    
    
def fill_feed_dict_with_some_ex(x, y_, data_images, data_labels, target_indices):
        input_feed = data_images[target_indices, :]
        labels_feed = data_labels[target_indices,:]
        feed_dict = {
            x: input_feed,
            y_: labels_feed,
        }
        return feed_dict 
    
def fill_feed_dict_with_batch(x, y_, Test_input, Test_label, batch_size=0):
        if batch_size is None:
            return fill_feed_dict_with_all_ex(x, y_, Test_input, Test_label)


def fill_feed_dict_with_all_ex(x, y_, data_images, data_labels):
        feed_dict = {
            x: data_images,
            y_: data_labels
        }
        return feed_dict
    

      
def get_inverse_hvp_lissa(v, sess, v_placeholder, hessian_vector,
                              batch_size=None,
                              scale=10, damping=0.0, num_samples=1, recursion_depth=10000):
        """
        This uses mini-batching; uncomment code for the single sample case.
        """    
        inverse_hvp = None
        print_iter = recursion_depth / 10

        for i in range(num_samples):
            # samples = np.random.choice(self.num_train_examples, size=recursion_depth)
           
            cur_estimate = v

            for j in range(recursion_depth):
             
                # feed_dict = fill_feed_dict_with_one_ex(
                #   data_set, 
                #   images_placeholder, 
                #   labels_placeholder, 
                #   samples[j])   
                feed_dict = fill_feed_dict_with_batch(x, y_, Test_input, Test_label, batch_size=batch_size)

                feed_dict = update_feed_dict_with_v_placeholder(v_placeholder, feed_dict, cur_estimate)
                hessian_vector_val = sess.run(hessian_vector, feed_dict=feed_dict)
                cur_estimate = [a + (1-damping) * b - c/scale for (a,b,c) in zip(v, cur_estimate, hessian_vector_val)]    

                # Update: v + (I - Hessian_at_x) * cur_estimate
                if (j % print_iter == 0) or (j == recursion_depth - 1):
                    print("Recursion at depth %s: norm is %.8lf" % (j, np.linalg.norm(cur_estimate[0])))
                    feed_dict = update_feed_dict_with_v_placeholder(v_placeholder, feed_dict, cur_estimate)

            if inverse_hvp is None:
                inverse_hvp = [b/scale for b in cur_estimate]
            else:
                inverse_hvp = [a + b/scale for (a, b) in zip(inverse_hvp, cur_estimate)]  

        inverse_hvp = [a/num_samples for a in inverse_hvp]
        return inverse_hvp        


def update_feed_dict_with_v_placeholder(v_placeholder, feed_dict, vec):
        for pl_block, vec_block in zip(v_placeholder, vec):
            feed_dict[pl_block] = vec_block        
        return feed_dict
    
    
    
## Define the Model and Path for Gradients

batch_size = 50
total_batch = int(mnist.train.num_examples/batch_size)
num_epochs = 5

tf.reset_default_graph()
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10], name="truth")

#Set the weights for the network
xavier = tf.contrib.layers.xavier_initializer_conv2d()  
conv1_weights = tf.get_variable(name="c1", initializer=xavier, shape=[5, 5, 1, 10])
conv1_biases = tf.Variable(tf.zeros([10]))
conv2_weights = tf.get_variable(name="c2", initializer=xavier, shape=[5, 5, 10, 25])
conv2_biases = tf.Variable(tf.zeros([25]))
conv3_weights = tf.get_variable(name="c3", initializer=xavier, shape=[4, 4, 25, 100])
conv3_biases = tf.Variable(tf.zeros([100]))
fc1_weights = tf.Variable(tf.truncated_normal([4 * 4 * 100, 10], stddev=0.1))
fc1_biases = tf.Variable(tf.zeros([10]))

#Stack the Layers
reshaped_input = tf.reshape(x, [-1, 28, 28, 1], name="absolute_input")
#layer 1
conv1 = tf.nn.conv2d(reshaped_input, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],  padding='SAME')
#layer 2
conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#layer 3
conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))
pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#layer 4    
pool_shape = pool3.get_shape().as_list()
reshaped = tf.reshape(pool3, [-1, pool_shape[1] * pool_shape[2] * pool_shape[3]])
y = tf.add(tf.matmul(reshaped, fc1_weights), fc1_biases, name="absolute_output")

# Define loss and optimizer
total_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

grads = tf.gradients(total_loss,x)

params = tf.trainable_variables()


grad_total_loss_op = tf.gradients(total_loss, params)
grad_loss_no_reg_op = grad_total_loss_op

v_placeholder = [tf.placeholder(tf.float32, shape=a.get_shape()) for a in params]
u_placeholder = [tf.placeholder(tf.float32, shape=a.get_shape()) for a in params]

hessian_vector = hessian_vector_product(total_loss, params, v_placeholder)

grad_loss_wrt_input_op = tf.gradients(total_loss, x)        

        # Because tf.gradients auto accumulates, we probably don't need the add_n (or even reduce_sum)        
influence_op = tf.add_n(
            [tf.reduce_sum(tf.multiply(a, array_ops.stop_gradient(b))) for a, b in zip(grad_total_loss_op, v_placeholder)])

grad_influence_wrt_input_op = tf.gradients(influence_op, x)
        
        
train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(total_loss)

########### Import Trained Model
saver = tf.train.Saver()
sess = tf.Session()

saver.restore(sess,"save_model/MNIST.ckpt") 


########### Test Indice is 34

Test_indices = [34]
test_grad_loss_no_reg_val = get_test_grad_loss_no_reg_val(sess, grad_loss_no_reg_op, Test_input, Test_label, Test_indices,batch_size=100 )

print('Norm of test gradient: %s' % np.linalg.norm(test_grad_loss_no_reg_val[0]))

inverse_hvp = get_inverse_hvp_lissa(test_grad_loss_no_reg_val, sess, v_placeholder, hessian_vector,
                              batch_size=None,
                              scale=10, damping=0.0, num_samples=1, recursion_depth=50)
        
        


########### Compute the Influence function 
num_to_remove = 1000
predicted_loss_diffs = np.zeros([num_to_remove])            
for counter in np.arange(num_to_remove):
                print(counter) 
                single_train_feed_dict = {x: Train_input[counter:counter+1, :], y_ : Train_label[counter:counter+1,:]}      
                train_grad_loss_val = sess.run(grad_total_loss_op, feed_dict=single_train_feed_dict)
                for q in np.arange(len(inverse_hvp)):
                    
                    predicted_loss_diffs[counter] = predicted_loss_diffs[counter] + np.dot(np.reshape(inverse_hvp[q],(1,-1)), np.reshape(train_grad_loss_val[q],(-1,1)))
                    
                predicted_loss_diffs[counter] = predicted_loss_diffs[counter] / num_to_remove          



#%%
plt.rcParams["font.family"] = "Times New Roman"
plt.figure()
plt.subplot(1,3,1)
fig = plt.imshow(np.reshape(Test_input[34,:], (28,28)))  
plt.title('Test Image')   

fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)


plt.subplot(1,3,2)
fig = plt.imshow(np.reshape(Train_input[0,:], (28,28)))  
plt.title('Harmful Image')   

fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)


plt.subplot(1,3,3)
fig = plt.imshow(np.reshape(Train_input[68,:], (28,28)))  
plt.title('Harmful Image')   

fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)

plt.savefig('InfluenceFunctionofDigits.png')
    