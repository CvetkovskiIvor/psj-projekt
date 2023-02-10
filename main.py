from randomForestRegressor import randomForestRegressor
from linearReg import linearRegression
from xgb import xgb

import matplotlib.pyplot as plt
from pandas import read_csv

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

data = read_csv(
        '/home/ivor/Downloads/housing.xls',
        header=None, delimiter=r"\s+", names=column_names)

randomForestRegressor()
linearRegression()

y_test_xgb, y_test_pred_xgb, y_train_xgb, y_pred_xgb = xgb(data)

# graficka usporedba algoritama na treniranju
plt.scatter(y_train_xgb, y_pred_xgb)
plt.scatter(y_train_xgb + 1, y_pred_xgb + 1)  # TODO zamijeniti plotanjem drugih algoritama
plt.title("Training data comparison")
plt.xlabel("MEDV")
plt.ylabel("Predicted MEDV")
plt.show()

# graficka usporedba algoritama na testiranju
plt.scatter(y_test_xgb, y_test_pred_xgb)
plt.scatter(y_test_xgb + 1, y_test_pred_xgb + 1)
plt.title("Testing data comparison")
plt.xlabel("MEDV")
plt.ylabel("Predicted MEDV")
plt.show()

print("finished executing code")