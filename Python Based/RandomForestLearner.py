#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 16:58:12 2019

@author: cankozan
"""

import pandas as pd
import numpy as np

from sklearn import datasets
data=datasets.load_iris()

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
x=data.data
y=data.target

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33)

rfc.fit(x_train,y_train)
predict=rfc.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,predict)
print(cm)

from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca.fit(data.data)
newdata=pca.transform(data.data)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(newdata,y,test_size=0.33)

rfc.fit(x_train,y_train)
predict=rfc.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,predict)
print(cm)