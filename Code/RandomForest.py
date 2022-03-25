#!/usr/bin/env python
# coding: utf-8

# In[9]:


#Import Statements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


# In[10]:


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


# In[11]:


#Training
rF = RandomForestClassifier(random_state=0, n_estimators=1000, oob_score=True, n_jobs=-1)
model = rF.fit(xtr, ytr)


# In[16]:


# Out of bag error calculation
oob_error = 1 - rF.oob_score_
oob_error


# In[17]:


# Testing
yhat_test = rF.predict(xtt)
#print(y_test.shape)


# In[20]:


# Confusion Matrix
CC_test = confusion_matrix(ytt, yhat_test.round())
pd.DataFrame(CC_test).to_csv('C:/Users/itali/Assignment1_410/ConfusionMatrix/Confusion02.csv')

TN = CC_test[0,0]
FP = CC_test[0,1]
FN = CC_test[1,0]
TP = CC_test[1,1]

FPFN = FP+FN
TPTN = TP+TN

# Calculated Measures
Accuracy = 1/(1+(FPFN/TPTN))
print("Calculated Accuracy:", Accuracy)
Precision = 1/(1+(FP/TP))
print("Calculated Precision:", Precision)

from sklearn import metrics
print("BuiltIn_Accuracy:",metrics.accuracy_score(ytt, yhat_test.round()))
print("BuiltIn_Sensitivity (recall):",metrics.recall_score(ytt, 
yhat_test.round(), average='weighted'))


# In[ ]:




