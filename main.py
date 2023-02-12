from randomForestRegressor import randomForestRegressor
from linearReg import linearRegression
from xgb import xgb

import matplotlib.pyplot as plt
from pandas import read_csv
import seaborn as sns

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

#data = read_csv(
 #       '/home/ivor/Downloads/housing.xls',
  #      header=None, delimiter=r"\s+", names=column_names)

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

y_test_xgb, y_test_pred_xgb, y_train_xgb, y_pred_xgb = xgb(data)
y_test_rfg, y_test_pred_rfg, y_train_rfg, y_pred_rfg =randomForestRegressor(data)
y_test_lr, y_test_pred_lr, y_train_lr, y_pred_lr =linearRegression(data)

# graficka usporedba algoritama na treniranju
plt.scatter(y_train_xgb, y_pred_xgb)
plt.scatter(y_train_rfg, y_pred_rfg)
plt.scatter(y_train_lr, y_pred_lr)
# TODO zamijeniti plotanjem drugih algoritama
plt.title("Training data comparison")
plt.xlabel("MEDV")
plt.ylabel("Predicted MEDV")
plt.show()

# graficka usporedba algoritama na testiranju
plt.scatter(y_test_xgb, y_test_pred_xgb)
plt.scatter(y_test_rfg, y_test_pred_rfg)
plt.scatter(y_test_lr, y_test_pred_lr)
plt.title("Testing data comparison")
plt.xlabel("MEDV")
plt.ylabel("Predicted MEDV")
plt.show()

print("finished executing code")
