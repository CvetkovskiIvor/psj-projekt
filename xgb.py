from xgboost import XGBRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pandas import read_csv
from sklearn.datasets import load_boston

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

data = read_csv(
    '/home/ivor/Downloads/housing.xls',
    header=None, delimiter=r"\s+", names=column_names)

# Provjera da li postoje redovi bez vrijednosti
print(data.isnull().sum())

print(data.describe())

# Prikaz korelacije između pojedinih varijabli sustava
correlation = data.corr().abs()
plt.figure(figsize=(20, 10))
sns.heatmap(correlation, annot=True)
plt.show()


boston = load_boston()

data = pd.DataFrame(boston.data)

data.head()

data.columns = boston.feature_names
data.head()

corr = data.corr()
corr.shape

plt.figure(figsize=(20,20))
sns.heatmap(corr, cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':15}, cmap='Greens')