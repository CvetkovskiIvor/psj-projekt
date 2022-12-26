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

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
#Skalira  vrijednosti u jedan jedinstveni range
min_max_scaler = preprocessing.MinMaxScaler()
#Odabrani su stupci koji imaju najveću korelaciju sa traženim stupcom
column_corr = ['LSTAT', 'INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX']
X = data.loc[:,column_corr]
X = min_max_scaler.fit_transform(X)
#X = data.drop(['MEDV'], axis=1)
# Tražena varijabla/stupac
y = data['MEDV']

# Podjela dataset-a na dva dijela. Jedan za trening, a jedan za testiranje
# Veličina dataset-a za test je 30% orginalne veličine testa
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=25)

print("Shape of X_train: ",X_train.shape)
print("Shape of X_test: ", X_test.shape)
print("Shape of y_train: ",y_train.shape)
print("Shape of y_test",y_test.shape)

# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
ForestReg = RandomForestRegressor()

# Training data
ForestReg.fit(X_train, y_train)
train_predicted = ForestReg.predict(X_train)
print("Predicted MEDV: ", train_predicted)
# Provjera ispravnosti algoritma
print("Accuracy of Random Forrest algorithm ", ForestReg.score(X_train,y_train))
# Računa koliko varijacije u dobivenom rezultatu se može predvidjeti na temelju ulazne varijable
# Što je broj bliže 1 to je algoritam točniji
print('R^2:',metrics.r2_score(y_train, train_predicted))
# Prosjek kvadrata razlike između dobivenih vrijednosti i stvarnih vrijednosti
print('MAE:',metrics.mean_absolute_error(y_train, train_predicted))
# Što je manji MSE to je greška manja
#https://datagy.io/mean-squared-error-python/ - dodatno objašnjenje
print('MSE:',metrics.mean_squared_error(y_train, train_predicted))
# https://www.kaggle.com/general/215997 - Objašnjenje za RMSE grešku
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train, train_predicted)))

plt.scatter(y_train, train_predicted)
plt.xlabel("MEDV")
plt.ylabel("Predicted MEDV")
plt.title("Training data: MEDV vs Predicted MEDV")
plt.show()


# Test data
print("---------------------------------------------------")
ForestReg.fit(X_test,y_test)
test_predicted = ForestReg.predict(X_test)
print("Accuracy of Random Forrest algorithm for test Data ", ForestReg.score(X_train,y_train))
print('R^2 Test:',metrics.r2_score(y_test, test_predicted))
print('MAE Test:',metrics.mean_absolute_error(y_test, test_predicted))
print('MSE Test:',metrics.mean_squared_error(y_test, test_predicted))
print('RMSE Test:',np.sqrt(metrics.mean_squared_error(y_test, test_predicted)))

plt.scatter(y_test, test_predicted)
plt.xlabel("MEDV")
plt.ylabel("Predicted MEDV")
plt.title("Test data: MEDV vs Predicted MEDV")
plt.show()