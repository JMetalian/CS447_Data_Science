#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 16:40:07 2019

@author: cankozan
"""

import numpy as np
import pandas as pd

# Read data from excel file
data = pd.read_excel('New_Data.xlsx')

result=pd.DataFrame()
for i in range(len(data)):
    if(data.iloc[i,0]>=0 and data.iloc[i,0]<220):
        if(data.iloc[i,1]>=0 and data.iloc[i,1]<220):
            result=pd.concat([result,data.iloc[i,:]],axis=1)
result=result.transpose()
result2= pd.concat([result.iloc[:,1:],result.iloc[:,:1]],axis=1)

x=result2.iloc[:,:2]
y=result2.iloc[:,2:]

#from sklearn.model_selection import train_test_split
#
#x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)
#
########################################################
#
#from sklearn.tree import DecisionTreeClassifier
#dtc= DecisionTreeClassifier()
#dtc.fit(x_train,y_train)
#y_pred=dtc.predict(x_test)
#print("Decision Tree")
#from sklearn.metrics import confusion_matrix
#cm=confusion_matrix(y_test,y_pred)
#print(cm)
#
########################################################
#
#from sklearn.naive_bayes import GaussianNB
#gnb=GaussianNB()
#gnb.fit(x_train,y_train)
#y_pred=gnb.predict(x_test)
#print("Naive Bayes")
#from sklearn.metrics import confusion_matrix
#cm=confusion_matrix(y_test,y_pred)
#print(cm)
#
########################################################
#
#from sklearn.neighbors import KNeighborsClassifier
#nn=KNeighborsClassifier()
#nn.fit(x_train,y_train)
#y_pred=nn.predict(x_test)
#print("KNN")
#from sklearn.metrics import confusion_matrix
#cm=confusion_matrix(y_test,y_pred)
#print(cm)
