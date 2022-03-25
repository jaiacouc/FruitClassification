#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[14]:


def data_split(fName):
    #Read a feature space and create label set
    input_data = pd.read_csv(fName, header=None)
    # Replace NaN values with mean values
    #input_data.fillna((input_data.mean()),inplace=True)

    NN = 65
    y = input_data[NN]
    
    # Drop Labels and store the features
    input_data.drop(NN,axis=1,inplace=True)
    X = input_data
    
    # Splitting 80:20
    xtr,xtt,ytr,ytt=train_test_split(X,y,test_size=0.2)
    
    #print(xtr)
    # To CSV
    pd.DataFrame(xtr).to_csv('C:/Users/itali/Assignment1_410/Training/xTraining01.csv', index=False)
    pd.DataFrame(ytr).to_csv('C:/Users/itali/Assignment1_410/Training/yTraining01.csv', index=False)
    pd.DataFrame(xtt).to_csv('C:/Users/itali/Assignment1_410/Testing/xTesting01.csv', index=False)
    pd.DataFrame(ytt).to_csv('C:/Users/itali/Assignment1_410/Testing/yTesting01.csv', index=False)
    
    #Training scatter plot
    fig = plt.figure(figsize = (10, 5))
    ax1 = fig.add_subplot(111)
    ax1.scatter(xtr[1], xtr[54], color = 'green', s = 1)
    plt.xlim(10, 250)
    plt.ylim(10, 250)
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 54')
    ax1.legend()
    plt.show()
    
    #Testing plot
    fig = plt.figure(figsize = (10, 5))
    ax1 = fig.add_subplot(111)
    ax1.scatter(xtt[1], xtt[54], color = 'green', s = 1)
    plt.xlim(10, 250)
    plt.ylim(10, 250)
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 54')
    ax1.legend()
    plt.show()
    
    plt.hist(xtr[[1,54]])
    plt.show()
    
    plt.hist(xtt[[1,54]])
    plt.show()


# In[15]:


data_split('C:/Users/itali/Assignment1_410/DataFrames/merged01.csv')


# In[ ]:




