import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pandas import read_csv



column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

data = read_csv('C:\\Users\\User\\Documents\\Fakultet\\Programiranje skriptni jezici\\psj-projekt\\Dataset\\housing.csv', header=None, delimiter=r"\s+", names=column_names)

#Provjera da li postoje redovi bez vrijednosti
print(data.isnull().sum())

print(data.describe())

#Prikaz korelacije između pojedinih varijabli sustava
correlation = data.corr().abs()
plt.figure(figsize=(20, 10))
sns.heatmap(correlation, annot=True)
plt.show()