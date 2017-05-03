from sklearn.utils import indexable
from sklearn.model_selection._split import check_cv
import numpy as np


def cross_testing(estimator, X_train, X_test, Y_train, Y_test, cv, scorer=None, model_params=None):
    """
    Parameters
    ----------
    estimator : estimator object.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a score function or scorer must be passed.

    X_train : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training vectors that are part of the training set, where n_samples is the
        number of samples and n_features is the number of features.

    X_test : array-like, shape (n_samples,)
        Target values that are part of the training set (class labels in classification,
        real numbers in regression)

    Y_train : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training vectors that are part of the test set, where n_samples is the number
        of samples and n_features is the number of features.

    Y_test : array-like, shape (n_samples,)
        Target values that are part of the test set (class labels in classification, real
        numbers in regression)

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy. Possible inputs for cv are:
            • None, to use the default 3-fold cross-validation,
            • Integer, to specify the number of folds in a (Stratified)KFold,
            • An object to be used as a cross-validation generator,
            • An iterable yielding train, test splits.
        For integer/None inputs, if the estimator is a classifier and the target is either
        binary or multiclass, StratifiedKFold is used. In all other cases, KFold is used.

    scorer : callable or None, default=None
        Scorer callable object / function with the signature scorer(esimator, X, y).
        If None, the score method of the estimator is used.

    model_params : dict, optional
        Hyper-parameters to pass to the estimator.

    Returns
    -------
    scores : list
        List of all the scores.

    Examples
    --------
    >>> from cross_testing import cross_testing
    >>> import numpy as np
    >>> from sklearn.svm import SVR
    >>> from sklearn.metrics import mean_squared_error
    >>> from sklearn.model_selection import GridSearchCV
    >>> n_samples, n_features = 10, 5
    >>> np.random.seed(0)
    >>> y_train = np.random.randn(n_samples)
    >>> y_test = np.random.randn(n_samples - 6)
    >>> x_train = np.random.randn(n_samples, n_features)
    >>> x_test = np.random.randn(n_samples - 6, n_features)
    >>> param_grid = {"C": [0.001, 0.01, 0.1, 1.0], "epsilon": [0.0, 0.1, 0.2, 0.3]}
    >>> grid_searched = GridSearchCV(SVR(), param_grid)
    >>> grid_searched.fit(x_train, y_train)
    >>> scores = cross_testing(grid_searched.best_estimator_, x_train, x_test, y_train, y_test, 3, \
    scorer=mean_squared_error, model_params=grid_searched.best_params_)
    """

    X_train, Y_train = indexable(X_train, Y_train)
    X_test, Y_test = indexable(X_test, Y_test)
    if model_params is not None:
        clf = estimator.set_params(model_params)
    else:
        clf = estimator
    cv = check_cv(cv)
    scores = []
    for X_test_index, Y_test_index in zip(cv.split(X_test), cv.split(Y_test)):
        X_train_new = np.vstack([X_train, X_test[X_test_index[1]]])
        Y_train_new = np.append(Y_train, Y_test[Y_test_index[1]])
        clf.fit(X_train_new, Y_train_new)
        scores.append(scorer(Y_test[Y_test_index[0]], clf.predict(X_test[X_test_index[0]])))
    return scores
