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
column_corr = ['LSTAT', 'INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX','AGE','DIS']
#X = data.drop(['MEDV'], axis=1)
X = data.loc[:,column_corr]
X = min_max_scaler.fit_transform(X)
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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score


n_estimators = [int(x) for x in np.arange(start = 10, stop = 2000, step = 10)]
max_features = [0.5,'auto', 'sqrt','log2']
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

#ForestReg =RandomForestRegressor(n_estimators=500, min_samples_leaf=1, max_features=0.5,bootstrap=False)
ForestReg = RandomForestRegressor(n_estimators=470,min_samples_leaf=1,max_features='sqrt',bootstrap=False)
ForestReg_random = RandomizedSearchCV(estimator = ForestReg, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

ForestReg_random.fit(X_train,y_train)
print(ForestReg_random.best_params_)
# Training data
ForestReg.fit(X_train, y_train)
train_predicted = ForestReg.predict(X_train)
print("Predicted MEDV: ", train_predicted)
# Provjera ispravnosti algoritma
print("Accuracy of Random Forrest algorithm ", ForestReg.score(X_train,y_train))
# Računa koliko varijacije u dobivenom rezultatu se može predvidjeti na temelju ulazne varijable
# Što je broj bliže 1 to je algoritam točniji
print('R^2:',metrics.r2_score(y_train, train_predicted))
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_train, train_predicted))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
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
plt.title("Random Forrest Training data: MEDV vs Predicted MEDV")
plt.show()


# Test data
print("---------------------------------------------------")
#ForestReg.fit(X_test,y_test)
test_predicted = ForestReg.predict(X_test)
print("Accuracy of Random Forrest algorithm for test Data ", ForestReg.score(X_train,y_train))
print('R^2 Test:',metrics.r2_score(y_test, test_predicted))
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_train, test_predicted))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
print('MAE Test:',metrics.mean_absolute_error(y_test, test_predicted))
print('MSE Test:',metrics.mean_squared_error(y_test, test_predicted))
print('RMSE Test:',np.sqrt(metrics.mean_squared_error(y_test, test_predicted)))

plt.scatter(y_test, test_predicted)
plt.xlabel("MEDV")
plt.ylabel("Predicted MEDV")
plt.title("Random Forrest Test data: MEDV vs Predicted MEDV")
plt.show()

#Linear regression
from sklearn.linear_model import LinearRegression

LinearReg = LinearRegression()

LinearReg.fit(X_train,y_train)

lmTrainPredict = LinearReg.predict(X_train)

print("-----------------------------------------------------------------")
print("Accuracy of Linear regression algorithm for train Data ", LinearReg.score(X_train,y_train))
print('Linear Regression train R^2:',metrics.r2_score(y_train, lmTrainPredict))
print('Linear Regression train Adjusted R^2:',1 - (1-metrics.r2_score(y_train, lmTrainPredict))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
print('Linear Regression train MAE:',metrics.mean_absolute_error(y_train, lmTrainPredict))
print('Linear Regression train MSE:',metrics.mean_squared_error(y_train, lmTrainPredict))
print('Linear Regression train RMSE:',np.sqrt(metrics.mean_squared_error(y_train, lmTrainPredict)))


plt.scatter(y_train, lmTrainPredict)
plt.xlabel("MEDV")
plt.ylabel("Predicted MEDV")
plt.title("Linear Regression Train data: MEDV vs Predicted MEDV")
plt.show()

#Test data
lmTestPredict = LinearReg.predict(X_test)

print("----------------------------------------------------------------------------")
print("Accuracy of Linear regression algorithm for test Data ", LinearReg.score(X_test,y_test))
print('Linear Regression test R^2:', metrics.r2_score(y_test, lmTestPredict))
print('Linear Regression test Adjusted R^2:',1 - (1-metrics.r2_score(y_train, lmTestPredict))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
print('Linear Regression test MAE:',metrics.mean_absolute_error(y_test, lmTestPredict))
print('Linear Regression test MSE:',metrics.mean_squared_error(y_test, lmTestPredict))
print('Linear Regression test RMSE:',np.sqrt(metrics.mean_squared_error(y_test, lmTestPredict)))


plt.scatter(y_test, lmTestPredict)
plt.xlabel("MEDV")
plt.ylabel("Predicted MEDV")
plt.title("Linear Regression TEST data: MEDV vs Predicted MEDV")
plt.show()