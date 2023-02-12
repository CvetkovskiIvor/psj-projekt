def svr(data):
    import numpy as np
    import matplotlib.pyplot as plt
    import time

    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    from sklearn import preprocessing

    # Provjera da li postoje redovi bez vrijednosti
    print(data.isnull().sum())
    print(data.describe())

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

    # Support Vector Regression
    from sklearn.svm import SVR
    from sklearn.model_selection import cross_val_score

    # SVRegr =SVR(n_estimators=500, min_samples_leaf=1, max_features=0.5,bootstrap=False)
    SVRegr = SVR(C=211.49654965532167, epsilon=0.1, degree=0.04127375231664331, gamma=1.2401627989937203)
    # SVRegr = SVR()
    # SVRegr_random = RandomizedSearchCV(estimator=SVRegr, param_distributions = param_distribs, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

    # SVRegr_random.fit(X_train, y_train)
    # print(SVRegr_random.best_params_)

    # timer start
    t1 = time.time()

    # Training data
    scores_map = {}
    scores = cross_val_score(SVRegr, X_train, y_train, cv=10)
    scores_map['Support Vector Regression'] = scores
    print("Mean score of %0.2f with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    SVRegr.fit(X_train, y_train)
    # print(SVRegr.get_params(deep=True))

    train_predicted = SVRegr.predict(X_train)
    print("Predicted MEDV: ", train_predicted)
    # Provjera ispravnosti algoritma
    print("Accuracy of Support Vector algorithm ", SVRegr.score(X_train,y_train))
    # Računa koliko varijacije u dobivenom rezultatu se može predvidjeti na temelju ulazne varijable
    # Što je broj bliže 1 to je algoritam točniji
    print('R^2:',metrics.r2_score(y_train, train_predicted))
    print('Adjusted R^2:',1 - (1-metrics.r2_score(y_train, train_predicted))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
    # Prosjek kvadrata razlike između dobivenih vrijednosti i stvarnih vrijednosti
    print('MAE:',metrics.mean_absolute_error(y_train, train_predicted))
    # Što je manji MSE to je greška manja
    # https://datagy.io/mean-squared-error-python/ - dodatno objašnjenje
    print('MSE:',metrics.mean_squared_error(y_train, train_predicted))
    # https://www.kaggle.com/general/215997 - Objašnjenje za RMSE grešku
    print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train, train_predicted)))

    plt.scatter(y_train, train_predicted)
    plt.xlabel("MEDV")
    plt.ylabel("Predicted MEDV")
    plt.title("Support Vector Training data: MEDV vs Predicted MEDV")
    plt.show()

    # Test data
    print("---------------------------------------------------")
    # SupportVec.fit(X_test,y_test)
    test_predicted = SVRegr.predict(X_test)

    # kraj timera
    t2 = time.time()
    duration = t2 - t1

    print("Accuracy of Support Vector algorithm for test Data ", SVRegr.score(X_train,y_train))
    print("Time of execution: ", t2 - t1,"secs")
    print('R^2 Test:',metrics.r2_score(y_test, test_predicted))
    print('Adjusted R^2:', 1 - (1-metrics.r2_score(y_test, test_predicted))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
    print('MAE Test:',metrics.mean_absolute_error(y_test, test_predicted))
    print('MSE Test:',metrics.mean_squared_error(y_test, test_predicted))
    print('RMSE Test:',np.sqrt(metrics.mean_squared_error(y_test, test_predicted)))

    """
    plt.scatter(y_test, test_predicted)
    plt.xlabel("MEDV")
    plt.ylabel("Predicted MEDV")
    plt.title("Support Vector Test data: MEDV vs Predicted MEDV")
    plt.show()

    plt.scatter(test_predicted,y_test-test_predicted)
    plt.title("Support Vector Reg Predicted vs residuals")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.show()

    print('------------------------------------------------------')

    plt.figure(figsize=(20, 10))
    scores_map = pd.DataFrame(scores_map)
    sns.boxplot(data=scores_map)
    plt.show()
    """
    return y_test, test_predicted, y_train, train_predicted, scores, duration
