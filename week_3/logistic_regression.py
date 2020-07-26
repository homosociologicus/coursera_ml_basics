import numpy as np

from pandas import read_csv, Series
from sklearn.metrics import roc_auc_score


# logistic regression with gradient descent, sigmoid activation function,
# 2 features in X, target y, initial weights w, step k, L2-regularization C,
# epsilon convergence eps and maximum iterations max_iter:
def logistic_regression(X, y, w=(0,0), k=0.1, C=0, eps=1e-5, max_iter=10000):
    w1, w2 = w
    x1 = Series(X.iloc[:, 0])
    x2 = Series(X.iloc[:, 1])

    # implementing gradient descent
    for _ in range(max_iter):
        w1_new = \
            w1 + k * np.mean(
                y * x1 * (1 - 1 / (1 + np.exp(- y * (w1 * x1 + w2 * x2))))
            ) - k * C * w1
        w2_new = \
            w2 + k * np.mean(
                y * x2 * (1 - 1 / (1 + np.exp(- y * (w1 * x1 + w2 * x2))))
            ) - k * C * w2

        # checking for convergence
        if (w1_new - w1) ** 2 + (w2_new - w2) ** 2 < eps ** 2:
            break

        w1, w2 = w1_new, w2_new

    # turning the result into estimated probabilities
    probabilities = 1 / (1 + np.exp(- w1_new * x1 - w2_new * x2))

    return probabilities


# loading data to test the function
data = read_csv('data-logistic.csv', header=None)
y = Series(data.iloc[:, 0])
X = data.iloc[:, 1:]

# observing results
print(f'auc_roc without regularization: {roc_auc_score(y, logistic_regression(X, y))}',
      f'auc_roc with regularization C=10: {roc_auc_score(y, logistic_regression(X, y, C=10))}',
      sep='\n')
