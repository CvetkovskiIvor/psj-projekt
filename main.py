import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pandas import read_csv

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

data = read_csv(
    'C:\\Users\\User\\Documents\\Fakultet\\Programiranje skriptni jezici\\psj-projekt\\Dataset\\housing.csv',
    header=None, delimiter=r"\s+", names=column_names)

# Provjera da li postoje redovi bez vrijednosti
print(data.isnull().sum())

print(data.describe())

# Prikaz korelacije između pojedinih varijabli sustava
correlation = data.corr().abs()
plt.figure(figsize=(20, 10))
sns.heatmap(correlation, annot=True)
plt.show()

# Podjela dataset-a na dva dijela. Jedan za trening, a jedan za testiranje
# Veličina dataset-a za test je 30% orginalne veličine testa
X = data.drop(['MEDV'], axis=1)
y = data['MEDV']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=25)

print("Shape of X_train: ",X_train.shape)
print("Shape of X_test: ", X_test.shape)
print("Shape of y_train: ",y_train.shape)
print("Shape of y_test",y_test.shape)

# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
ForestReg = RandomForestRegressor()
ForestReg.fit(X_train, y_train)
predicted_result = ForestReg.predict(X_train)
print("Predicted MEDV: ", predicted_result)
# Basic funkcija za provjeru ispravnosti algoritma.
print("Accuracy of Random Forrest algorithm ", ForestReg.score(X_train,y_train))
