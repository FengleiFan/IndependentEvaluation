import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pickle


### Loading CIFAR100
def load_cifar100(filename):

    with open(filename, 'rb')as f:
        datadict = pickle.load(f, encoding='latin1')
        images = datadict['data']
        labels = datadict['fine_labels']
        labels = np.array(labels)
                
        return images, labels
    
images_train, labels_train = load_cifar100("./cifar-100-python/train")
images_test, labels_test = load_cifar100("./cifar-100-python/test")


### The function to draw weights

def draw_weights(synapses, Kx, Ky):
    yy=np.random.randint(90, size=1)
    HM=np.zeros((32*Ky,32*Kx,3))
    for y in range(Ky):
        for x in range(Kx):
            HM[y*32:(y+1)*32,x*32:(x+1)*32,:]=synapses[yy,:].reshape(32,32,3)
            yy += 1
    plt.clf()
    nc=np.amax(np.absolute(HM))
    im=plt.imshow(HM,cmap='bwr',vmin=-nc,vmax=nc)
    fig.colorbar(im,ticks=[np.amin(HM), 0, np.amax(HM)])
    plt.axis('off')
    fig.canvas.draw()
    
   
### Model hyperparamters

M = images_train
Nc=10
N=3072
Ns=50000

eps0=2e-2    # learning rate
Kx=10
Ky=10
hid=1000    # number of hidden units that are displayed in Ky by Kx array
mu=0.0
sigma=1.0
Nep=200      # number of epochs
Num=100      # size of the minibatch
prec=1e-30
delta=0.4    # Strength of the anti-hebbian learning
p=2.0        # Lebesgue norm of the weights
k=2          # ranking parameter, must be integer that is bigger or equal than 2

### Bio-learning

fig=plt.figure(figsize=(12.9,10))

synapses = np.random.normal(mu, sigma, (hid, N))
for nep in range(Nep):
    print(nep)
    eps=eps0*(1-nep/Nep)
    M=M[np.random.permutation(Ns),:]
    for i in range(Ns//Num):
        inputs=np.transpose(M[i*Num:(i+1)*Num,:])
        sig=np.sign(synapses)
        tot_input=np.dot(sig*np.absolute(synapses)**(p-1),inputs)
        
        y=np.argsort(tot_input,axis=0)
        yl=np.zeros((hid,Num))
        yl[y[hid-1,:],np.arange(Num)]=1.0
        yl[y[hid-k],np.arange(Num)]=-delta
        
        xx=np.sum(np.multiply(yl,tot_input),1)
        ds=np.dot(yl,np.transpose(inputs)) - np.multiply(np.tile(xx.reshape(xx.shape[0],1),(1,N)),synapses)
        
        nc=np.amax(np.absolute(ds))
        if nc<prec:
            nc=prec
        synapses += eps*np.true_divide(ds,nc)
        
    draw_weights(synapses, Kx, Ky)
    

### Visualization


Kx = 20
Ky = 50

fig=plt.figure(figsize=(12.9,10))    
yy=0

HMI=np.zeros((32*Ky,32*Kx,3))
for y in range(Ky):
    for x in range(Kx):
        
        HM = synapses[yy]
        
        HM_max = np.max(HM)
        HM_min = np.min(HM)
        
        HM = (HM-HM_min)/(HM_max-HM_min)
        
        HM = np.reshape(HM,(3,32,32)).transpose((1,2,0))        
        
        HMI[y*32:(y+1)*32,x*32:(x+1)*32,:]= HM
        yy += 1
plt.clf()
nc=np.amax(np.absolute(HMI))
im=plt.imshow(HMI,cmap='bwr',vmin=-nc,vmax=nc)
fig.colorbar(im,ticks=[np.amin(HMI), 0, np.amax(HMI)])
plt.axis('off')
fig.canvas.draw()



### Save Weights
synapse_transpose = np.transpose(synapses)
np.save('FrozenWeights_cifar100.npy', synapse_transpose)

