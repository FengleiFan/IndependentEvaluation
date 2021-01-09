import numpy as np
import numpy.random as npr
import tensorflow as tf


from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow.contrib.eager as tfe
from main_neural_ode import NeuralODE


### tf.enable_eager_execution must be called at program startup. Please restart your kernel.


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 18})


keras = tf.keras
tfe.enable_eager_execution()

### Initialize parameters

t = np.linspace(0, 25, 200)
h0 = tf.to_float([[1., 0.]])
W = tf.to_float([[-0.1, 1.0], [-0.2, -0.1]])

### Define the Computational Graph

class Lambda(tf.keras.Model):
    def call(self, inputs, **kwargs):
        t, h = inputs
        return tf.matmul(h, W)
    
neural_ode = NeuralODE(Lambda(), t=t)
hN, states_history = neural_ode.forward(h0, return_states="numpy")
initial_path = np.concatenate(states_history)   

### This is a function to plot the trajectory

def plot_trajectory(trajectories, fig=True):
    if fig:
        plt.figure(figsize=(5, 5))
        
    for path in trajectories:
        if type(path) == tuple:
            c, label, path = path
            plt.plot(*path.T, c, lw=2, label=label)
        else:
            plt.plot(*path.T, lw=2)
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    

plot_trajectory([initial_path]) 

### Define model parameters

optimizer = tf.train.MomentumOptimizer(learning_rate=1e-2, momentum=0.95)
h0_var = tf.contrib.eager.Variable(h0)
hN_target = tf.to_float([[0., 0.5]])

### compute the gradient with respect to the h0 and W

def compute_gradients_and_update():    
    with tf.GradientTape() as g:       
        hN = neural_ode.forward(h0_var)
        g.watch(hN)
        loss = tf.reduce_sum((hN_target - hN)**2)
        
    dLoss = g.gradient(loss, hN) # same what 2 * (hN_target - hN)
    h0_reconstruction, dfdh0, dWeights = neural_ode.backward(hN, dLoss)

    optimizer.apply_gradients(zip([dfdh0], [h0_var]))

    return loss


### Compile EAGER graph to static (this will be much faster)

compute_gradients_and_update = tfe.defun(compute_gradients_and_update)


### Show the Optimization Process

loss_history = []

for step in tqdm(range(201)):
   with tf.GradientTape() as g:       
        hN = neural_ode.forward(h0_var)
        g.watch(hN)
        loss = tf.reduce_sum((hN_target - hN)**2)
   dLoss = g.gradient(loss, hN) # same what 2 * (hN_target - hN)
   h0_reconstruction, dfdh0, dWeights = neural_ode.backward(hN, dLoss)
   print(dWeights) 
   optimizer.apply_gradients(zip([dfdh0], [h0_var]))
   
   if step % 50 == 0:        
        yN, states_history_model = neural_ode.forward(h0_var, return_states="numpy")    
        plot_trajectory([
            ("r", "initial", initial_path), 
            ("g", "optimized", np.concatenate(states_history_model))])        
        plt.show()
        
print(dfdh0)    
print(h0_var)    


    
