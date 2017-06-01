# Cross-validation & cross-testing

Cross-validation & cross-testing is a module for scikit-learn that adds a function to perform cross-testing. Cross-testing is where the testing set is being iterated through in a way similar to cross-validation, so that all of the testing data has been used to train the model at least once.

The cross-validation & cross-testing method was developed by Korjus et al.

## Getting Started

These instructions will get the module working with your local installation of scikit-learn.

### Prerequisites

Since this is a module for scikit-learn, it will require:
* Python (>= 3.3)
* NumPy (>= 1.6.1)
* SciPy (>= 0.9)
* Scikit-learn (>= 0.18)

### Installing

Getting started is very easy. First you must download the package and then simply import into your current project.

```
from cross_testing import cross_testing
```


## Using the module

The cross_testing function is to be used to test the parameters that cross-validation has chosen for the machine learning model. In this example we will randomly generate the data, but the course of action is similar to using it on real data.

First import the necessary prerequisites.

```
from cross_testing import cross_testing
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
```

Next we generate the necessary data in the required form. y_train is the target data in the training set. y_test is the target data in the testing set. x_train holds the features and the data that predictions are to be made on in the training set. y_train is the same for the testing set.

```
n_samples, n_features = 10, 5
np.random.seed(0)
y_train = np.random.randn(n_samples)
y_test = np.random.randn(n_samples - 6)
x_train = np.random.randn(n_samples, n_features)
x_test = np.random.randn(n_samples - 6, n_features)
```

We will write the parameter grid, which is from where cross-validation will choose the different values for the named hyper-parameters. We will also do grid search with GridSearchCV and will be using support vector regression. In the last two lines we will be getting the scores from the cross-testing function that was developed in this module.

```
param_grid = {"C": [0.001, 0.01, 0.1, 1.0], "epsilon": [0.0, 0.1, 0.2, 0.3]}
grid_searched = GridSearchCV(SVR(), param_grid)
grid_searched.fit(x_train, y_train)
scores = cross_testing(grid_searched.best_estimator_, x_train, x_test, y_train, y_test, 3,
    scorer=mean_squared_error, model_params=grid_searched.best_params_)
```


## License

This project is licensed under the BSD 3-clause license - see the [LICENSE.md](LICENSE.md) file for details