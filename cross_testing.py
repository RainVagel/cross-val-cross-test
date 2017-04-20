from sklearn.metrics import accuracy_score
from sklearn.utils import indexable
from sklearn.model_selection._split import check_cv
# from _split import check_cv
import numpy as np


def cross_testing(estimator, X_train, X_test, Y_train, Y_test, cv, scorer, model_params=None):
    X_train, Y_train = indexable(X_train, Y_train)
    X_test, Y_test = indexable(X_test, Y_test)
    if model_params is not None:
        clf = estimator.set_params(model_params)
    else:
        clf = estimator
    cv = check_cv(cv)
    accuracies = []
    for X_test_index, Y_test_index in zip(cv.split(X_test), cv.split(Y_test)):
        X_train_new = np.vstack([X_train, X_test[X_test_index[1]]])
        Y_train_new = np.append(Y_train, Y_test[Y_test_index[1]])
        clf.fit(X_train_new, Y_train_new)
        accuracies.append(scorer(Y_test[Y_test_index[0]], clf.predict(X_test[X_test_index[0]])))
    return accuracies
