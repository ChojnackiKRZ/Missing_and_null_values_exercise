# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 10:56:47 2022

@author: krzys
"""
"""
Zadania wstępne:
-wczytaj plik csv
-wyświetl liczbę (tablica + bar plot z %) brakujących wartości per cecha (kolumna)
-sprawdź typy danych
-sprawdź czy są duplikaty (opcjonalnie)
-wyznacz korelacje cech vs kolumna 'Price'
-utwórz histogramy (3 na jednym plocie) dla różnej liczby pokoi vs cena 
(sprawdź ile jest mieszkań dla danej liczby pokoi i wyznacz reprezentatywne przedziały) 
    - opcjonalnie
"""
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


pd.options.display.float_format = "{:.2%}".format

path = r"C:\Users\krzys\Desktop\data science\IV semestr\machine_learning\Missing_and_null_values_exercise"
os.chdir(path)

df = pd.read_csv("houses_data.csv")

# locate columns with nulls and count
df_null = df[df.columns[df.isna().any()]]
null_count = df_null.isnull().sum().rename("null_count").to_frame()
null_count["pct_empty"] = null_count["null_count"] / len(df_null)
print(null_count)

columns = df_null.columns.tolist()

for col in columns:
    df_null[col + "_nan"] = ""
# done with for-loop because logic mask did not work for np.nans
for col in columns:
    col_name = col + "_nan"
    for record in range(0, len(df_null)):
        if str(df_null[col].iloc[record]) == "nan":
            df_null[col_name].iloc[record] = True
# bar plot showing null count with percentage above each bar
plt.figure(figsize=(8, 8))
graph = plt.bar(null_count.index, null_count["null_count"])
plt.title("Number and percentage of nulls in columns with null")

i = 0
for p in graph:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    plt.text(
        x + width / 2,
        y + height * 1.01,
        str(round(null_count["pct_empty"][i] * 100, 2)) + "%",
        ha="center",
        weight="bold",
    )
    i += 1
plt.show()

# datatypes and numeric columns selection for correlation
df.dtypes
numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
num_df = df.select_dtypes(include=numerics)

# duplicates
if df.duplicated(keep="last").iloc[0] == False:
    print("No duplicates")
else:
    print("Duplicates present")
# correlation
# values between 0.3 and -0.3 were hidden on heatmap to increase readability
corr_matrix = num_df.corr()
corr_matrix_logic_mask = (corr_matrix < 0.3) | (corr_matrix < -0.3)
sns.heatmap(corr_matrix, annot=True, mask=corr_matrix_logic_mask)
plt.show()
corr_price = corr_matrix["Price"].rename("Price_corr").to_frame()

#%%
"""Zadanie główne:

1. Wczytaj ponownie plik (utwórz funkcję), gdzie:

będą pobierane tylko wybrane kolumny,
zostaną zdefiniowane typy ww. kolumn.
"""
import pandas as pd
from typing import List


def load_file(file_name: str, dtypes: List[str] = None, columns: List[str] = "All") -> pd.DataFrame:
    """
    Allows to load a file in selected directory.
    Parameter columns needs a list of strings. Default value is 'All', which
    means whole file will be loaded.
    Function returns 2 data frames: first for loaded file, second consisting of
    columns with nulls.
    """
    global null_cols
    file_name = str(file_name)
    if columns == "All":
        df = pd.read_csv(file_name)
        null_cols = df.columns[df.isna().any()].rename("col_name")
        return pd.read_csv(file_name)
    else:
        if type(columns) != list:
            raise TypeError("columns parameter must be a list of strings")
        elif dtypes != None and type(dtypes) != list:
            raise TypeError("dtypes must be a list of strings")
        else:
            for types in columns:
                if type(types) != str:
                    raise TypeError("each value in list must be a string")
        dictionary = dict(zip(columns, dtypes))
        df = pd.read_csv(file_name, usecols=columns)
        null_cols = df.columns[df.isna().any()].rename("col_name")
        return pd.read_csv(file_name, usecols = columns, dtype = dictionary)


df = load_file("houses_data.csv")
#%%
"""Uzupełnianie brakujących danych"""
"""1.SimpleImputer"""
from sklearn.impute import SimpleImputer
import numpy as np

# numerical columns
numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
num_df = df.select_dtypes(include=numerics)
num_cols = num_df.columns

# SimpleImputer
# for numerical
median_imputer = SimpleImputer(missing_values=np.nan, strategy="median")
median_imputer = median_imputer.fit(df[["Car", "BuildingArea"]])
imputed_df_median = pd.DataFrame(
    median_imputer.transform(df[["Car", "BuildingArea"]])
).rename(columns={0: "SI_Car", 1: "SI_BuildingArea"})

# for categorical
mode_imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
mode_imputer = mode_imputer.fit(df[["CouncilArea"]])
imputed_df_mode = pd.DataFrame(mode_imputer.transform(df[["CouncilArea"]])).rename(
    columns={0: "SI_CouncilArea"}
)

# for time?

"""2.Pandas"""
median_pd_nans = pd.DataFrame()
for col in ["Car", "BuildingArea"]:
    median_pd_nans["pd_" + col] = df[col].fillna(df[col].median())
# bfill -> do tyłu, ffil -> do przodu
pd_bfill_ffill = (
    df_null["CouncilArea"].bfill().ffill().rename("pd_CouncilArea").to_frame()
)

"""3.Key-nearest-neighbours"""
from sklearn.impute import KNNImputer

# mapping for KNN for categorical values
uniqe_CouncilArea = dict(enumerate(df_null["CouncilArea"].unique().flatten()))
# reverse key-value for mapper
uniqe_CouncilArea = {v: k for k, v in uniqe_CouncilArea.items()}
impt = KNNImputer(missing_values=np.nan)
# uniqe values for dict
df_null["CouncilAreaID"] = df_null["CouncilArea"].map(uniqe_CouncilArea)
impt = impt.fit(df_null[["Car", "BuildingArea", "CouncilAreaID"]])
impt_results = pd.DataFrame(
    impt.transform(df_null[["Car", "BuildingArea", "CouncilAreaID"]])
)
impt_results = impt_results.rename(
    columns={0: "KNN_Car", 1: "KNN_BuildingArea", 2: "KNN_CouncilAreaID"}
)
# re-mapping
impt_results["KNN_CouncilArea"] = impt_results["KNN_CouncilAreaID"].map(
    {v: k for k, v in uniqe_CouncilArea.items()}
)

"""Zestawienie wyników"""
data_frames = [
    df_null,
    imputed_df_median,
    imputed_df_mode,
    median_pd_nans,
    pd_bfill_ffill,
    impt_results,
]
summary_df = pd.concat(data_frames, axis=1)


#%%
# train-test-split
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
houses_predictors = df[["Rooms", "Bedroom2"]]
houses_target = df["Price"]


X_train, X_test, y_train, y_test = train_test_split(
    houses_predictors, houses_target, train_size=0.7, test_size=0.3, random_state=0
)


def score_dataset(X_train, X_test, y_train, y_test):
    regr_model = LinearRegression()
    regr_model.fit(X_train, y_train)
    preds = regr_model.predict(X_test)
    return mean_absolute_error(y_test, preds)


MAE = score_dataset(X_train, X_test, y_train, y_test)
