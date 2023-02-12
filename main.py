from randomForestRegressor import randomForestRegressor
from linearReg import linearRegression
from xgb import xgb
from SVR import svr

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pandas import read_csv

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

data = read_csv(
       '/home/ivor/Downloads/housing.xls',
       header=None, delimiter=r"\s+", names=column_names)

# data = read_csv(
#        'C:\\Users\\User\\Documents\\Fakultet\\Programiranje skriptni jezici\\psj-projekt\\Dataset\\housing.csv',
#        header=None, delimiter=r"\s+", names=column_names)

# Provjera da li postoje redovi bez vrijednosti
print(data.isnull().sum())
print(data.describe())

# Prikaz korelacije između pojedinih varijabli sustava
correlation = data.corr().abs()
plt.figure(figsize=(20, 10))
sns.heatmap(correlation, annot=True)
plt.show()

# pohranjivanje rezultata algoritama u varijable
y_test_xgb, y_test_pred_xgb, y_train_xgb, y_pred_xgb, scores_xgb, duration_xgb = xgb(data)
y_test_rfg, y_test_pred_rfg, y_train_rfg, y_pred_rfg, scores_rfg, duration_rfg = randomForestRegressor(data)
y_test_lr, y_test_pred_lr, y_train_lr, y_pred_lr, scores_lr, duration_lr = linearRegression(data)
y_test_svr, y_test_pred_svr, y_train_svr, y_pred_svr, scores_svr, duration_svr = svr(data)

# pohrana vrijednosti preciznosti algoritama u data frame
scores_map = {}
scores_map['Linear Regression'] = scores_lr
scores_map['RFG'] = scores_rfg
scores_map['XGB'] = scores_xgb
scores_map['SVR'] = scores_svr
 
# graficka usporedba preciznosti
scores_map = pd.DataFrame(scores_map)
sns.boxplot(data=scores_map)
plt.title("Accuracy comparison")
plt.show()

# graficka usporedba algoritama na treniranju
plt.scatter(y_train_xgb, y_pred_xgb, label='XGB')
plt.scatter(y_train_rfg, y_pred_rfg, label='Random Forrest Regressor')
plt.scatter(y_train_lr, y_pred_lr, label='Linear regression')
plt.scatter(y_train_svr, y_pred_svr, label='SVR')
# TODO zamijeniti plotanjem drugih algoritama
plt.title("Training data comparison")
plt.xlabel("MEDV")
plt.ylabel("Predicted MEDV")
plt.legend()
plt.show()

# graficka usporedba algoritama na testiranju
plt.scatter(y_test_xgb, y_test_pred_xgb, label='XGB')
plt.scatter(y_test_rfg, y_test_pred_rfg, label='Random Forrest Regressor')
plt.scatter(y_test_lr, y_test_pred_lr, label='Linear regression')
plt.scatter(y_test_svr, y_test_pred_svr, label='SVR')
# TODO zamijeniti plotanjem drugih algoritama
plt.title("Testing data comparison")
plt.xlabel("MEDV")
plt.ylabel("Predicted MEDV")
plt.legend()
plt.show()

plt.bar(['linear regression', 'random forrest regressor', 'XGB', 'SVR'], [duration_lr, duration_rfg, duration_xgb, duration_svr])
plt.title("Execution time comparison")
plt.ylabel("Time in seconds")
plt.show()

print(duration_lr, duration_rfg, duration_xgb, duration_svr)
print("finished executing code")
