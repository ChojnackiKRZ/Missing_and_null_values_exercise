# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 10:56:47 2022

@author: krzys
"""

import pandas as pd
import os
import matplotlib.pyplot as plt

pd.options.display.float_format = '{:.2%}'.format

path = r"C:\Users\krzys\Desktop\data science\IV semestr\machine_learning\Missing_and_null_values_exercise"
os.chdir(path)

df = pd.read_csv('houses_data.csv')

#locate columns with nulls and count
df_null = df[df.columns[df.isna().any()]]
null_count = df_null.isnull().sum().rename('null_count').to_frame()
null_count['pct_empty'] = null_count['null_count'] / len (df_null)
print (null_count)

#bar plot showing null count
plt.figure(figsize = (8, 8))
graph = plt.bar(null_count.index, null_count['null_count'])
plt.title('Number and percentage of nulls in columns with null')

i = 0
for p in graph:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    plt.text(x+width/2,
             y+height*1.01,
             str(round (null_count['pct_empty'][i] * 100, 2))+'%',
             ha='center',
             weight='bold')
    i+=1
plt.show()
