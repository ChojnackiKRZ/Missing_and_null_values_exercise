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

columns = df_null.columns.tolist()

for col in columns:
    df_null[col + '_nan'] = ''

#done with for-loop because logic mask did not work for np.nans
for col in columns:
    col_name = col + '_nan'
    for record in range (0, len (df_null)):
        if str(df_null[col].iloc[record]) == 'nan':
            df_null[col_name].iloc[record] = True
            
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

#duplicates
if df.duplicated(keep = 'last').iloc[0] == False:
    print ('No duplicates')
else:
    print ('Duplicates present')

#correlation
#values between 0.3 and -0.3 were hidden on heatmap to increase readability
corr_matrix = num_df.corr()
corr_matrix_logic_mask = ((corr_matrix < 0.3) | (corr_matrix < -0.3))
sns.heatmap(corr_matrix, annot=True, mask = corr_matrix_logic_mask)
plt.show()
corr_price = corr_matrix['Price'].rename('Price_corr').to_frame()

#%%
'''Zadanie 2'''
import pandas as pd
from typing import List

def load_file (file_name: str, columns: List[str] = 'All') -> pd.DataFrame:
    '''
    Allows to load a file in selected directory.
    Parameter columns needs a list of strings. Default value is 'All', which
    means whole file will be loaded.
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
'''Uzupełnianie brakujących danych'''
'''1.SimpleImputer'''
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
imputed_df_median = pd.DataFrame(median_imputer.transform(df[['Car', 'BuildingArea']])).rename(columns = {0 : 'SI_Car', 1 : 'SI_BuildingArea'})

#for categorical
mode_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
mode_imputer = mode_imputer.fit(df[['CouncilArea']])
imputed_df_mode = pd.DataFrame(mode_imputer.transform(df[['CouncilArea']])).rename(columns = {0 : 'SI_CouncilArea'})

#for time?

'''2.Pandas'''
median_pd_nans = pd.DataFrame()
for col in ['Car', 'BuildingArea']:
    median_pd_nans['pd_' + col] = df[col].fillna(df[col].median())

#bfill -> do tyłu, ffil -> do przodu
pd_bfill_ffill = df_null['CouncilArea'].bfill().ffill().rename('pd_CouncilArea').to_frame()

'''3.Key-nearest-neighbours'''
from sklearn.impute import KNNImputer
from functools import reduce
#mapping for KNN for categorical values
uniqe_CouncilArea = dict (enumerate (df_null['CouncilArea'].unique().flatten()))
#reverse key-value for mapper
uniqe_CouncilArea = {v: k for k, v in uniqe_CouncilArea.items()}
impt = KNNImputer(missing_values=np.nan)
#uniqe values for dict
df_null['CouncilAreaID'] = df_null['CouncilArea'].map(uniqe_CouncilArea)
impt = impt.fit(df_null[['Car', 'BuildingArea', 'CouncilAreaID']])
impt_results = pd.DataFrame(impt.transform(df_null[['Car', 'BuildingArea', 'CouncilAreaID']]))
impt_results = impt_results.rename(columns = {0 : 'KNN_Car', 1 : 'KNN_BuildingArea', 2 : 'KNN_CouncilAreaID'})
#re-mapping
impt_results['KNN_CouncilArea'] = impt_results['KNN_CouncilAreaID'].map({v: k for k, v in uniqe_CouncilArea.items()})

'''Zestawienie wyników'''
data_frames = [df_null, imputed_df_median, imputed_df_mode, median_pd_nans, pd_bfill_ffill, impt_results]
summary_df = pd.concat(data_frames, axis=1)

#%%
'''Usuwanie brakujacych danych'''
'''1.Usuwanie wartosci pustych przy pomocy pandas'''
df_nulls = df[null_cols]

drop_axis1 = df_nulls.dropna(axis = 1) #usuwa kolumny z nullami
drop_axis0 = df_nulls.dropna(axis = 0) #usuwa wiersze z nullami
drop_all = df_nulls.dropna(how = 'all') #usuwa wiersze z nullami tylko tam
                                        #gdzie cały wiersz jest nullem
drop_thresh = df_nulls.dropna(thresh = 2) #Keep only the rows with at least 
                                          #x non-NA values
drop_col = df_nulls.dropna(subset = ['Car']) #usuwa tylko nulle z danej kolumny
                                             #wywala jednak caly wiersz

