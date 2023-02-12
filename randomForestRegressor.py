def randomForestRegressor(data):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import time
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    from sklearn import preprocessing


    # Skalira  vrijednost i u jedan jedinstveni range
    min_max_scaler = preprocessing.MinMaxScaler()
    # Odabrani su stupci koji imaju najveću korelaciju sa traženim stupcom
    column_corr = ['LSTAT', 'INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX', 'AGE', 'DIS']
    X = data.loc[:, column_corr]
    X = min_max_scaler.fit_transform(X)
    # Tražena varijabla/stupac
    y = data['MEDV']

    # Podjela dataset-a na dva dijela. Jedan za trening, a jedan za testiranje
    # Veličina dataset-a za test je 30% orginalne veličine testa
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=25)

    print("Shape of X_train: ", X_train.shape)
    print("Shape of X_test: ", X_test.shape)
    print("Shape of y_train: ", y_train.shape)
    print("Shape of y_test", y_test.shape)

    # Random Forest Regressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.model_selection import cross_val_score

    n_estimators = [int(x) for x in np.arange(start=10, stop=2000, step=10)]
    max_features = [0.5, 'auto', 'sqrt', 'log2']
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    # ForestReg =RandomForestRegressor(n_estimators=500, min_samples_leaf=1, max_features=0.5,bootstrap=False)
    ForestReg = RandomForestRegressor(n_estimators=470, min_samples_leaf=1, max_features='sqrt', bootstrap=False)
    # ForestReg_random = RandomizedSearchCV(estimator = ForestReg, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

    # ForestReg_random.fit(X_train,y_train)
    # print(ForestReg_random.best_params_)
    # Training data

    start_time = time.time()

    ForestReg.fit(X_train, y_train)

    scores_map = {}
    scores = cross_val_score(ForestReg, X_train, y_train, cv=10)
    scores_map['Random Forrest regressor'] = scores
    print("Mean score of %0.2f with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

    train_predicted = ForestReg.predict(X_train)
    print("Predicted MEDV: ", train_predicted)
    # Provjera ispravnosti algoritma
    print("Accuracy of Random Forrest algorithm ", ForestReg.score(X_train, y_train))
    # Računa koliko varijacije u dobivenom rezultatu se može predvidjeti na temelju ulazne varijable
    # Što je broj bliže 1 to je algoritam točniji
    print('R^2:', metrics.r2_score(y_train, train_predicted))
    print('Adjusted R^2:',
          1 - (1 - metrics.r2_score(y_train, train_predicted)) * (len(y_train) - 1) / (len(y_train) - X_train.shape[1] - 1))
    # Prosjek kvadrata razlike između dobivenih vrijednosti i stvarnih vrijednosti
    print('MAE:', metrics.mean_absolute_error(y_train, train_predicted))
    # Što je manji MSE to je greška manja
    # https://datagy.io/mean-squared-error-python/ - dodatno objašnjenje
    print('MSE:', metrics.mean_squared_error(y_train, train_predicted))
    # https://www.kaggle.com/general/215997 - Objašnjenje za RMSE grešku
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, train_predicted)))

    plt.scatter(y_train, train_predicted)
    plt.xlabel("MEDV")
    plt.ylabel("Predicted MEDV")
    plt.title("Random Forrest Training data: MEDV vs Predicted MEDV")
    plt.show()

    end_time = time.time()
    print("Execution time Random Forrest: ", end_time - start_time, "secs")

    # Test data
    print("---------------------------------------------------")
    # ForestReg.fit(X_test,y_test)
    test_predicted = ForestReg.predict(X_test)
    print("Accuracy of Random Forrest algorithm for test Data ", ForestReg.score(X_train, y_train))
    print('R^2 Test:', metrics.r2_score(y_test, test_predicted))
    print('Adjusted R^2:',
          1 - (1 - metrics.r2_score(y_test, test_predicted)) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1))
    print('MAE Test:', metrics.mean_absolute_error(y_test, test_predicted))
    print('MSE Test:', metrics.mean_squared_error(y_test, test_predicted))
    print('RMSE Test:', np.sqrt(metrics.mean_squared_error(y_test, test_predicted)))

    plt.scatter(y_test, test_predicted)
    plt.xlabel("MEDV")
    plt.ylabel("Predicted MEDV")
    plt.title("Random Forrest Test data: MEDV vs Predicted MEDV")
    plt.show()

    """
    plt.scatter(test_predicted, y_test - test_predicted)
    plt.title("Random Forrest Reg Predicted vs residuals")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.show()
    """

    duration = end_time - start_time

    return y_test, test_predicted, y_train, train_predicted, scores, duration


