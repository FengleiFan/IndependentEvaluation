print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.neural_network import MLPRegressor
from sklearn.datasets.california_housing import fetch_california_housing

import math
from itertools import combinations
import shap


## Use House Dataset
cal_housing = fetch_california_housing()

X, y = cal_housing.data, cal_housing.target
names = cal_housing.feature_names

# Center target to avoid gradient boosting init bias: gradient boosting
# with the 'recursion' method does not account for the initial estimator
# (here the average target, by default)
y -= y.mean()

## Train a MLP regressor
print("Training MLPRegressor...")
est = MLPRegressor(activation='logistic')
est.fit(X, y)

y_ = est.predict(X)


## derive Shapley values based on its definition

Index = [0,1,2,3,4,5,6,7]
ShapleyValue = np.zeros(X.shape)

for i in np.arange(X.shape[0]):

      Instance_i =X[i:i+1,:]
      print(i)

      for j in np.arange(X.shape[1]):
          
         Index_delete_j = np.delete(Index,[j])

         
         for S in np.arange(len(Index_delete_j)):
            List = list(combinations(Index_delete_j,S+1))
          
            for NumberOfCombination in np.arange(len(List)):
                TypeOfCombination = np.array(List[NumberOfCombination])
                
                Instance_i_with_j = np.zeros((1,8))
                Instance_i_with_j[:,:] = Instance_i
                

                Instance_i_with_j[:,TypeOfCombination] = 0
                Instance_i_without_j = np.zeros((1,8))
                Instance_i_without_j[:,:] = Instance_i_with_j
                Instance_i_without_j[:,j] = 0

                
                cmod = S+1
                Factorials_concerned = float(math.factorial(cmod) * math.factorial(8 - cmod - 1)) / float(math.factorial(8))
                
                ShapleyValue[i,j] = ShapleyValue[i,j]+ Factorials_concerned*(est.predict(Instance_i_with_j)-est.predict(Instance_i_without_j))
                
                

## Use Library Shap to plot

import shap
import pandas as pd  

X_data, y_data = cal_housing.data, cal_housing.target   

e_dataframe = pd.DataFrame(X_data)      

new_data = e_dataframe.rename(index=str, columns={0:'MedInc'})     
new_data = new_data.rename(index=str, columns={1:'HouseAge'})  
new_data = new_data.rename(index=str, columns={2:'AveRooms'})  
new_data = new_data.rename(index=str, columns={3:'AveBedrms'})  
new_data = new_data.rename(index=str, columns={4:'Population'})  
new_data = new_data.rename(index=str, columns={5:'AveOccup'})  
new_data = new_data.rename(index=str, columns={6:'Latitude'})  
new_data = new_data.rename(index=str, columns={7:'Longitude '})  
    
shap.summary_plot(ShapleyValue, new_data)

 
    
