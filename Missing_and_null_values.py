# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 10:56:47 2022

@author: krzys
"""
'''Zadanie 1'''
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

pd.options.display.float_format = '{:.2%}'.format

path = r"C:\Users\krzys\Desktop\data science\IV semestr\machine_learning\Missing_and_null_values_exercise"
os.chdir(path)

df = pd.read_csv('houses_data.csv')

#locate columns with nulls and count
df_null = df[df.columns[df.isna().any()]]
null_count = df_null.isnull().sum().rename('null_count').to_frame()
null_count['pct_empty'] = null_count['null_count'] / len (df_null)
print (null_count)

#bar plot showing null count with percentage above each bar
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

#datatypes and numeric columns selection for correlation
df.dtypes
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
num_df = df.select_dtypes(include=numerics)

#correlation
corr_matrix = num_df.corr()
sns.heatmap(corr_matrix, annot=True)
plt.show()
corr_price = corr_matrix['Price'].rename('Price_corr').to_frame()


#%%
'''Zadanie 2'''
import pandas as pd
from typing import List

def load_file (file_name: str, columns: List[str] = 'All') -> pd.DataFrame:
    '''
    Allows to load a file in selected directory.
    Parameter columns need list of strings. Default value is 'All', which
    means on default whole file will be loaded.
    Function returns 2 data frames: first for loaded file, second consisting of
    columns with nulls.
    '''
    global null_cols
    file_name = str(file_name)
    if columns == 'All':
        df = pd.read_csv(file_name)
        null_cols = df.columns[df.isna().any()].rename('col_name')
        return pd.read_csv(file_name)
    else:
        if type(columns) != list:
            raise TypeError('columns parameter must be a list of strings')
        else:
            for types in columns:
                if type(types) != str:
                    raise TypeError('each value in list must be a string')
        df = pd.read_csv(file_name, usecols=columns)
        null_cols = df.columns[df.isna().any()].rename('col_name')
        return pd.read_csv(file_name, usecols=columns)

df = load_file('houses_data.csv')
#%%
from sklearn.impute import SimpleImputer
import numpy as np

#numerical columns
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
num_df = df.select_dtypes(include=numerics)
num_cols = num_df.columns

#SimpleImputer
#for numerical
median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
median_imputer = median_imputer.fit(df[['Car', 'BuildingArea']])
imputed_df = pd.DataFrame(median_imputer.transform(df[['Car', 'BuildingArea']]))

#for categorical
mode_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
mode_imputer = mode_imputer.fit(df[['CouncilArea']])
imputed_df_2 = pd.DataFrame(mode_imputer.transform(df[['CouncilArea']]))

#for time?


#%%

