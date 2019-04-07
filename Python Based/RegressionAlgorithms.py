#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 17:22:28 2019

@author: cankozan
"""
import pandas as pd
import numpy as np

data = pd.read_csv('XU100.IS.csv')
data = data.dropna()

t1 = pd.to_datetime('03/26/2019')
for i in range(len(data)):
    date = pd.to_datetime(data.iloc[i,0])
    diff = (t1-date).days
    data.iloc[i,6]=1826-diff

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
xx = pd.DataFrame(data['Volume'])
yy = pd.DataFrame(data['Close'])
lr.fit(X = xx, y = yy )
predictions = pd.DataFrame(np.array([2000,2100,3100]),columns=['Volume'])
pred = lr.predict(predictions)
print(pred)