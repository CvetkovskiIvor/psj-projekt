def linearRegression(data):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import time

    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    from sklearn import preprocessing
    from sklearn.model_selection import cross_val_score

    from sklearn.linear_model import LinearRegression


    # Skalira  vrijednost i u jedan jedinstveni range
    min_max_scaler = preprocessing.MinMaxScaler()
    # Odabrani su stupci koji imaju najveću korelaciju sa traženim stupcom
    column_corr = ['LSTAT', 'INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX', 'AGE', 'DIS']
    # X = data.drop(['MEDV'], axis=1)
    X = data.loc[:, column_corr]
    X = min_max_scaler.fit_transform(X)
    # Tražena varijabla/stupac
    y = data['MEDV']

    # Podjela dataset-a na dva dijela. Jedan za trening, a jedan za testiranje
    # Veličina dataset-a za test je 30% orginalne veličine testa
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=25)

    scores_map = {}

    LinearReg = LinearRegression()

    start_timeLin = time.time()

    LinearReg.fit(X_train, y_train)

    lmTrainPredict = LinearReg.predict(X_train)

    print("-----------------------------------------------------------------")
    scores = cross_val_score(LinearReg, X_train, y_train, cv=10)
    scores_map['Linear Regression'] = scores
    print("Mean score of %0.2f with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    print("Accuracy of Linear regression algorithm for train Data ", LinearReg.score(X_train, y_train))
    print('Linear Regression train R^2:', metrics.r2_score(y_train, lmTrainPredict))
    print('Linear Regression train Adjusted R^2:',
          1 - (1 - metrics.r2_score(y_train, lmTrainPredict)) * (len(y_train) - 1) / (len(y_train) - X_train.shape[1] - 1))
    print('Linear Regression train MAE:', metrics.mean_absolute_error(y_train, lmTrainPredict))
    print('Linear Regression train MSE:', metrics.mean_squared_error(y_train, lmTrainPredict))
    print('Linear Regression train RMSE:', np.sqrt(metrics.mean_squared_error(y_train, lmTrainPredict)))

    plt.scatter(y_train, lmTrainPredict)
    plt.xlabel("MEDV")
    plt.ylabel("Predicted MEDV")
    plt.title("Linear Regression Train data: MEDV vs Predicted MEDV")
    plt.show()

    end_timeLin = time.time()
    print("Execution time Linear Regression: ", end_timeLin - start_timeLin, "secs")

    # Test data
    lmTestPredict = LinearReg.predict(X_test)

    print("----------------------------------------------------------------------------")
    print("Accuracy of Linear regression algorithm for test Data ", LinearReg.score(X_test, y_test))
    print('Linear Regression test R^2:', metrics.r2_score(y_test, lmTestPredict))
    print('Linear Regression test Adjusted R^2:',
          1 - (1 - metrics.r2_score(y_test, lmTestPredict)) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1))
    print('Linear Regression test MAE:', metrics.mean_absolute_error(y_test, lmTestPredict))
    print('Linear Regression test MSE:', metrics.mean_squared_error(y_test, lmTestPredict))
    print('Linear Regression test RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lmTestPredict)))

    plt.scatter(y_test, lmTestPredict)
    plt.xlabel("MEDV")
    plt.ylabel("Predicted MEDV")
    plt.title("Linear Regression TEST data: MEDV vs Predicted MEDV")
    plt.show()

    plt.scatter(lmTestPredict, y_test - lmTestPredict)
    plt.title("Predicted vs residuals")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.show()

    # plt.figure(figsize=(20, 10))
    # scores_map = pd.DataFrame(scores_map)
    # sns.boxplot(data=scores_map)
    # plt.show()

    return y_test, lmTestPredict, y_train, lmTrainPredict