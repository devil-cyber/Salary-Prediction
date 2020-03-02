#!/usr/bin/env python
# coding: utf-8

# In[3]:


# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 13:15:23 2020

@author: manikant
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:,:-1]
y=dataset.iloc[:,1]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)


# In[5]:


X_train


# In[10]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)


# In[11]:


y_pred=regressor.predict(X_test)


# In[12]:


y_pred


# In[19]:


plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('salary vs experince (training set)')
plt.xlabel('Years of Experince')
plt.ylabel('salary')
plt.show()


# In[20]:


plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('salary vs experince (Test set)')
plt.xlabel('Years of Experince')
plt.ylabel('salary')
plt.show()


# In[ ]:




