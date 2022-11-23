import pandas as pd

income = pd.read_csv("/Users/utente/desktop/famiglie/data we need/rper20.csv")
covariates = pd.read_csv("/Users/utente/desktop/famiglie/data we need/carcom20.csv")
heritage = pd.read_csv("/Users/utente/desktop/famiglie/data we need/ricfam20.csv")

income = income[["NQUEST", "nord", "Y"]]
heritage = heritage[["NQUEST", "W"]]

income.shape
covariates.shape

data = pd.merge(income, covariates, on=['NQUEST', 'nord'])
data = pd.merge(data, heritage, on=['NQUEST'])

data.shape
data.columns
