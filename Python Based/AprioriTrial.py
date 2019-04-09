#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 17:16:16 2019

@author: cankozan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv('AprioriData.csv',header=None)

t=[]
for i in range(0,10):
    t.append([str(data.values[i,j]) for j in range(0,6)])
from apyori import apriori
rules=apriori(t,min_support=0.1,min_confidence=0.2,min_lift=3,min_length=2)
print(list(rules))
    