from xgboost import XGBRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

data = read_csv(
    '/home/ivor/Downloads/housing.xls',
    header=None, delimiter=r"\s+", names=column_names)

# Provjera da li postoje redovi bez vrijednosti
print(data.isnull().sum())

print(data.describe())

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

reg = XGBRegressor()

reg.fit(X_train, y_train)

y_pred = reg.predict(X_train)

print('R^2:',metrics.r2_score(y_train, y_pred))
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_train, y_pred))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_train, y_pred))
print('MSE:',metrics.mean_squared_error(y_train, y_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train, y_pred)))

plt.scatter(y_train, y_pred)
plt.title("XGB Prices vs Predicted prices")
plt.xlabel("Prices")
plt.ylabel("Predicted prices")
plt.show()

plt.scatter(y_pred,y_train-y_pred)
plt.title("XGB Predicted vs residuals")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.show()

y_test_pred = reg.predict(X_test)

acc_xgb = metrics.r2_score(y_test, y_test_pred)
print('R^2:', acc_xgb)
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_test, y_test_pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_test, y_test_pred))
print('MSE:',metrics.mean_squared_error(y_test, y_test_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))