#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 17:16:08 2019

@author: cankozan
"""
import pandas as pd
import numpy as np

df=pd.read_excel('data.xlsx')
df=df.dropna()

from sklearn.cluster import KMeans

km=KMeans(n_clusters=6)

km.fit(df.iloc[:,:2])
labeled=km.predict(df.iloc[:,:2])


from sklearn.cluster import AgglomerativeClustering
ac=AgglomerativeClustering(n_clusters=6)
labeled_ac= ac.fit_predict(df.iloc[:,:2])

from scipy.cluster.hierarchy import dendrogram, linkage
z= linkage(df.iloc[:,:2])
dendrogram(z)



from sklearn.cluster import DBSCAN

dbs=DBSCAN(eps=10, min_samples=2)
labeled_dbs=dbs.fit_predict(df.iloc[:,:2])