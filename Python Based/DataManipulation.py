# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd

# Read data from excel file
data = pd.read_excel('New_Data.xlsx')

# Select the gender column
x = data.iloc[:, 2]

# Select rows 2 to 6 from data
x2 = data.iloc[2:6, :]

# Column filtering
fc = data.iloc[:, :1]
sc = data.iloc[:, 2:]

filteredColumn = pd.concat([fc, sc], axis=1)

# Row filtering
fp = data.iloc[:3, :]
sp = data.iloc[4:, :]

filteredRow = pd.concat([fp, sp], axis=1)

result = pd.DataFrame()
for i in range(len(data)):
    if(data.iloc[i, 0] % 2 == 0):
        result = pd.concat([result, data.iloc[i,:]], axis=1)
result = result.transpose()


bmi = []
for i in range(len(data)):
    bmi.append(data.iloc[i,1]/(data.iloc[i,0]/100*data.iloc[i,0]/100))

bmi_df = pd.DataFrame(bmi, columns=['BMI'])
data_with_bmi = pd.concat([data, bmi_df], axis=1)

bmi_info = []

for i in range(len(data_with_bmi)):
    if data_with_bmi.iloc[i, 3] < 60:
        bmi_info.append('thin')
    elif data_with_bmi.iloc[i, 3] < 80:
        bmi_info.append('normal')
    else:
        bmi_info.append('fat')
        
bmi_info_df = pd.DataFrame(bmi_info, columns=['BMI INFO'])        
data_with_bmi_info = pd.concat([data_with_bmi, bmi_info_df], axis=1)

from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
mms.fit(data.iloc[:, :2])
normalized_data = mms.transform(data.iloc[:, :2]) 

grouped_data = data.groupby(['gender']).mean()
