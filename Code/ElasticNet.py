#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import statements
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
#Elastic Net Regression Model
from sklearn.linear_model import ElasticNet


# In[2]:


# Training with 80% data
#X1_train = X1[0:TR-1,:]
#Y1_train = Y1[0:TR-1]

# Import Data
xtr = pd.read_csv('C:/Users/itali/Assignment1_410/Training/xTraining01.csv', header=None, index_col=False)
ytr = pd.read_csv('C:/Users/itali/Assignment1_410/Training/yTraining01.csv', header=None, index_col=False)
xtt = pd.read_csv('C:/Users/itali/Assignment1_410/Testing/xTesting01.csv', header=None, index_col=False)
ytt = pd.read_csv('C:/Users/itali/Assignment1_410/Testing/yTesting01.csv', header=None, index_col=False)

# Replace NaN values with the mean
xtr.fillna((xtr.mean()),inplace=True)
ytr.fillna((ytr.mean()),inplace=True)
xtt.fillna((xtt.mean()),inplace=True)
ytt.fillna((ytt.mean()),inplace=True)


# In[3]:


# Train model
en = ElasticNet(random_state=0)
en.fit(xtr,ytr)


# In[4]:


# Testing with 20% data
yhat_test = en.predict(xtt)
#print(ytt.shape)


# In[15]:


# Create confusion matrix and save to a csv
CC_test = confusion_matrix(ytt, yhat_test.round())
pd.DataFrame(CC_test).to_csv('C:/Users/itali/Assignment1_410/ConfusionMatrix/Confusion01.csv')
#print(CC_test)

TN = CC_test[0,0]
FP = CC_test[0,1]
FN = CC_test[1,0]
TP = CC_test[1,1]

FPFN = FP+FN
TPTN = TP+TN

# Confusion Analytics
Accuracy = 1/(1+(FPFN/TPTN))
print("Our_Accuracy_Score:",Accuracy)
Precision = 1/(1+(FP/TP))
print("Our_Precision_Score:",Precision)

from sklearn import metrics
print("BuiltIn_Accuracy:",metrics.accuracy_score(ytt, yhat_test.round()))
print("BuiltIn_Sensitivity (recall):",metrics.recall_score(ytt, 
yhat_test.round(), average='micro'))


# In[ ]:




