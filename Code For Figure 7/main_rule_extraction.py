import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets

import numpy as np

import tensorflow as tf

### Import Iris Data to play with

# Sepal Length, Sepal Width, Petal Length and Petal Width
iris = datasets.load_iris()
X = iris.data  # we only take the first two features.
y_temp = iris.target


Y = np.zeros((150,3))
for k in np.arange(150):
    Y[k,y_temp[k]] = 1


### Import a MLP Model

tf.reset_default_graph()
# correct labels
y_ = tf.placeholder(tf.float32, [None, 3])

# input data
x = tf.placeholder(tf.float32, [None, 4])

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.001)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

W_fc1 = weight_variable([4, 3])
b_fc1 = bias_variable([3])
h_fc1 = tf.nn.sigmoid(tf.matmul(x, W_fc1) + b_fc1)

W_fc2 = weight_variable([3, 3])
b_fc2 = bias_variable([3])
h_fc2 = tf.nn.sigmoid(tf.matmul(h_fc1, W_fc2) + b_fc2)

y = tf.nn.softmax(h_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.9).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

### Import Trained Model

saver = tf.train.Saver()
sess = tf.Session()       
saver.restore(sess,"SaveIrisModel/iris.ckpt")    
train_accuracy = sess.run(accuracy, feed_dict={x: X, y_ : Y})
        
print(train_accuracy) 

### Find Rules

# set epsilon
train_accuracy = sess.run(accuracy, feed_dict={x: X, y_ : Y})
epsilon = 0.5

activations = sess.run(h_fc1, feed_dict={x : X, y_ : Y} )

W_fc = sess.run(W_fc2)
b_fc = sess.run(b_fc2)

List_First_Hidden_Unit_1 = []
List_First_Hidden_Unit_2 = []
  
Average_First_Hidden_1 = 0
Average_First_Hidden_2 = 0
 
for k in np.arange(150):
    
      if  np.abs(activations[k,0]-0) < epsilon :
          List_First_Hidden_Unit_1.append([k])
          Average_First_Hidden_1 = Average_First_Hidden_1 + activations[k,0]
      else :
          List_First_Hidden_Unit_2.append([k])
          Average_First_Hidden_2 = Average_First_Hidden_2 + activations[k,0]

Average_First_Hidden_1 = Average_First_Hidden_1/len(List_First_Hidden_Unit_1)
Average_First_Hidden_2 = Average_First_Hidden_2/len(List_First_Hidden_Unit_2)

Hidden_1_index = np.array(List_First_Hidden_Unit_1)
Hidden_2_index = np.array(List_First_Hidden_Unit_2)

          
List_Second_Hidden_Unit_1 = []
List_Second_Hidden_Unit_2 = []
Average_Second_Hidden_1 = 0
Average_Second_Hidden_2 = 0
    
for k in np.arange(150):
    
      if  np.abs(activations[k,1]-0) < epsilon :
          List_Second_Hidden_Unit_1.append([k])
          Average_Second_Hidden_1 = Average_Second_Hidden_1 + activations[k,1]
      else :
          List_Second_Hidden_Unit_2.append([k])       
          Average_Second_Hidden_2 = Average_Second_Hidden_2 + activations[k,1]
          
Average_Second_Hidden_1 = Average_Second_Hidden_1/len(List_Second_Hidden_Unit_1)
Average_Second_Hidden_2 = Average_Second_Hidden_2/len(List_Second_Hidden_Unit_2)
          
List_Third_Hidden_Unit_1 = []
List_Third_Hidden_Unit_2 = []
Average_Third_Hidden_1 = 0
Average_Third_Hidden_2 = 0
    
for k in np.arange(150):
    
      if  np.abs(activations[k,2]-0) < epsilon :
          List_Third_Hidden_Unit_1.append([k])
          Average_Third_Hidden_1 = Average_Third_Hidden_1 + activations[k,2]
      else :
          List_Third_Hidden_Unit_2.append([k])       
          Average_Third_Hidden_2 = Average_Third_Hidden_2 + activations[k,2]
          
Average_Third_Hidden_1 = Average_Third_Hidden_1/len(List_Third_Hidden_Unit_1)
Average_Third_Hidden_2 = Average_Third_Hidden_2/len(List_Third_Hidden_Unit_2)          
      
# is equivalent to a discretized 
a = np.reshape(np.arange(8, dtype=np.uint8),(8,1))

b = np.unpackbits(a, axis=1)

b=b[:,5:8]

print(b)

index_x,index_y1 = np.where(b.T==1)
index_x,index_y2 = np.where(b.T==0)

new_activation = np.zeros((8,3))

new_activation[index_y1[0:4],0] = Average_First_Hidden_1
new_activation[index_y2[0:4],0] = Average_First_Hidden_2

new_activation[index_y1[4:8],1] = Average_Second_Hidden_1
new_activation[index_y2[4:8],1] = Average_Second_Hidden_2

new_activation[index_y1[8:12],2] = Average_Third_Hidden_1
new_activation[index_y2[8:12],2] = Average_Third_Hidden_2


digits = np.dot(new_activation,W_fc)+b_fc
Group_0 = []
Group_1 = []
Group_2 = []
Group_3 = []
Group_4 = []
Group_5 = []
Group_6 = []
Group_7 = []

for i in np.arange(150):
    if [i] in List_First_Hidden_Unit_1 :    
        
        if [i] in List_Second_Hidden_Unit_1 :   
            
            if [i] in List_Third_Hidden_Unit_1 :                
                Group_0.append(i)                
            else: 
                Group_1.append(i)
                
        else:
            if [i] in List_Third_Hidden_Unit_1 :                
                Group_2.append(i)                
            else: 
                Group_3.append(i)            
    else:
        
        if [i] in  List_Second_Hidden_Unit_1 :            
            if [i] in  List_Third_Hidden_Unit_1 :                
                Group_4.append(i)                
            else: 
                Group_5.append(i)
                
        else:
            if [i] in List_Third_Hidden_Unit_1 :                
                Group_6.append(i)                
            else: 
                Group_7.append(i)      

Group_0 = tuple(Group_0)
Group_1 = tuple(Group_1)
Group_2 = tuple(Group_2)
Group_3 = tuple(Group_3)
Group_4 = tuple(Group_4)
Group_5 = tuple(Group_5)
Group_6 = tuple(Group_6)
Group_7 = tuple(Group_7) 

for j in np.arange(4):
    
  print('The third class', j, np.max(X[Group_0,j]))  # the third class
  print('The third class', j, np.min(X[Group_0,j]))

  
  print('The second class', j, np.max(X[Group_1,j]))  # the second class
  print('The second class', j, np.min(X[Group_1,j]))
  
  print('The first class', j, np.max(X[Group_7,j]))  # The first class
  print('The first class', j, np.min(X[Group_7,j]))


### Evaluate the Effectiveness of Rules

# The rule sets:
# if petal length < 1.9: Iris Petosa 1
# if  petal length > 3.0 & petal width < 1.4 : Versticutor 2
# default Verginica 3 

Label_by_rule = np.zeros((150,1))

for j in np.arange(150):
    
    if X[j,2] < 1.9 :
        Label_by_rule[j,0] = 0
    elif (X[j,2] > 3.0) &  (X[j,3] < 1.7) & (X[j,2] < 5.0):
        Label_by_rule[j,0] = 1
    else:
        Label_by_rule[j,0] = 2
summm = 0
for j in np.arange(150):
    if Label_by_rule[j,0] ==  y_temp[j]:
        summm +=1
    
print(summm/150)
  