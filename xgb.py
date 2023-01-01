import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
from sklearn import preprocessing

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

data = read_csv(
    '/home/ivor/Downloads/housing.xls',
    header=None, delimiter=r"\s+", names=column_names)

# Provjera da li postoje redovi bez vrijednosti
print(data.isnull().sum())

print(data.describe())

#Odabrani su stupci koji imaju najveću korelaciju sa traženim stupcom
column_corr = ['LSTAT', 'INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX']

#Skalira  vrijednosti u jedan jedinstveni range
#min_max_scaler = preprocessing.MinMaxScaler()

X = data.loc[:,column_corr]
# Tražena varijabla/stupac
y = data.iloc[:,-1]

# Podjela dataset-a na dva dijela. Jedan za trening, a jedan za testiranje
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


xg_reg = xgb.XGBRegressor(objective="reg:squarederror",
                          learning_rate=0.1,
                          n_estimators=1000,
                          colsample_bytree=0.3,
                          max_depth=4,
                          alpha=10,
                          random_state=42)

print(xg_reg)

xg_reg.fit(X_train, y_train)

xg_score = xg_reg.score(X_train, y_train)

print("Training score: ", xg_score)

scores = cross_val_score(xg_reg, X_train, y_train,cv=10)
print("Mean cross-validation score: %.2f" % scores.mean())

# Model prediction on train data
y_pred = xg_reg.predict(X_train)

# Model evaluation
print('R^2:',metrics.r2_score(y_train, y_pred))
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_train, y_pred))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_train, y_pred))
print('MSE:',metrics.mean_squared_error(y_train, y_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train, y_pred)))

plt.scatter(y_train, y_pred)
plt.title("XGB training data: MEDV vs Predicted MEDV")
plt.xlabel("MEDV")
plt.ylabel("Predicted MEDV")
plt.show()

plt.scatter(y_pred,y_train-y_pred)
plt.title("Predicted vs residuals")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.show()

y_test_pred = xg_reg.predict(X_test)

print("------------------")
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
rmse

plt.scatter(y_test, y_test_pred)
plt.title("XGB test data: MEDV vs Predicted MEDV")
plt.xlabel("MEDV")
plt.ylabel("Predicted MEDV")
plt.show()

acc_xgb = metrics.r2_score(y_test, y_test_pred)
print('R^2:', acc_xgb)
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_test, y_test_pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_test, y_test_pred))
print('MSE:',metrics.mean_squared_error(y_test, y_test_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))