

import pandas as pd
import numpy


income = pd.read_csv("/Users/utente/desktop/famiglie/data we need/rper20.csv")
covariates = pd.read_csv("/Users/utente/desktop/famiglie/data we need/carcom20.csv")
heritage = pd.read_csv("/Users/utente/desktop/famiglie/data we need/ricfam20.csv")
 
income=income[["NQUEST", "nord","Y"]] 
heritage=heritage[["NQUEST","W"]] 

income.shape
covariates.shape

data=pd.merge(income, covariates, on=['NQUEST', 'nord'])
data=pd.merge(data, heritage, on=['NQUEST'])

data.shape
data.columns=="NQUEST"

a=data.isnull().sum()
for i in a:
    print(i)

data=data[["NQUEST", 'nord', 'Y','NCOMP', 'SEX', 'NASC', 'ANASC', 'CIT'
        , 'STACIV', 'STUDIO', 'TIPOLAU', 'VOTOEDU', 'SUEDU',
       'SELODE', 'TIPODIP', 'NESPLAV', 'DISLAV','CONTRIB', 'OCCNOW', 'PROBLAV1',
        'SALMIN',
       'SMARTW', 'ETA','QUAL',
       'Q', 'SETT', 'CFRED', 'PERC', 'NPERC',  'ACOM5', 'PESOFIT', 'PESOFIT2', 'W']] 


data = data.fillna(0)



['NQUEST', 'nord', 'Y', 'NCOMP', 'SEX', 'NASC', 'ANASC', 'CIT', 'STACIV',
       'STUDIO', 'TIPOLAU', 'VOTOEDU', 'SUEDU', 'SELODE', 'TIPODIP', 'NESPLAV',
       'DISLAV', 'CONTRIB', 'OCCNOW', 'PROBLAV1', 'SALMIN', 'SMARTW', 'ETA',
       'QUAL', 'Q', 'SETT', 'CFRED', 'PERC', 'NPERC', 'ACOM5', 'PESOFIT',
       'PESOFIT2', 'W']
[ False, False, False, False, True, True, False, True, False,
       True, True, False, False, False, True, False, 
       True, True,True, False, False, True, False, 
       True, True, True, False,True, False, True, False,
       False, False]

#################################### One hot encoding

def getfullitemsforOHE(train_test_combined,categorical_columns):
    fulllist=[]
    for feat in categorical_columns:
        fulllist.append(train_test_combined[feat].unique())
    return fulllist

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

from sklearn.preprocessing import OneHotEncoder

idx_cat = [ False, False, False, False, True, True, False, True, False,
       True, True, False, False, False, True, False, 
       True, True,True, False, False, True, False, 
       True, True, True, False,True, False, True, False,
       False, False]

idx_num= [not a for a in idx_cat]

categorical_columns = data.columns[idx_cat].tolist()
data = data.applymap(int)
data = data.applymap(str)


numerical_columns = data.columns[idx_num].tolist()

X_train_cat = data.filter(categorical_columns, axis=1)
X_train_cat = X_train_cat.applymap(int)
X_train_cat = X_train_cat.applymap(str)

X_train_num = data.filter(numerical_columns, axis=1)


cats=getfullitemsforOHE(data,categorical_columns)#

ohe = OneHotEncoder(categories=cats, sparse=False,handle_unknown="error",drop='first')
X_train_cat_transformed=ohe.fit_transform(X_train_cat[categorical_columns])
#let's present them in a dataframe
X_train_cat_transformed=pd.DataFrame(X_train_cat_transformed,columns=ohe.get_feature_names(categorical_columns))

X_train = pd.concat([X_train_num, X_train_cat_transformed], axis=1)

##########



final_data=X_train
final_data.to_csv("italy_income.csv", index=False) 

