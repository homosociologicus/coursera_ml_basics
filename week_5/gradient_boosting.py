import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import log_loss


# columns: 0 is activity, others are features
data = pd.read_csv('gbm-data.csv')
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, 1:].values,
                                                    data.iloc[:, 0].values,
                                                    test_size=0.8,
                                                    random_state=241)

# with 250 boosting stages studying log-loss for different learning rates
for rate in [1, 0.5, 0.3, 0.2, 0.1]:

    # training the model
    clf = GradientBoostingClassifier(
        learning_rate=rate,
        n_estimators=250,
        verbose=True,
        random_state=241
    )
    clf.fit(X_train, y_train)

    # initializing lists for losses
    test_loss = []
    train_loss = []

    # filling them with values from stages of model's decision-making
    # using log_loss between true class and sigmoid of predicted
    for y_pred in clf.staged_decision_function(X_train):
        train_loss.append(log_loss(y_train, 1 / (1 + np.exp(-y_pred))))
    for y_pred in clf.staged_decision_function(X_test):
        test_loss.append(log_loss(y_test, 1 / (1 + np.exp(-y_pred))))

    # observing minimum losses for each rate
    print(f'''
for learning_rate={rate}
train loss: min, iteration are {min(train_loss)}, {np.argmin(train_loss)}
test loss: min, iteration are {min(test_loss)}, {np.argmin(test_loss)}
''')

    # plotting losses without blocking the loop
    plt.figure()
    plt.title(f'Train and test losses for learning_rate={rate}')
    plt.xlabel('Iteration')
    plt.ylabel('Log_loss')
    plt.plot(test_loss, 'r', linewidth=2)
    plt.plot(train_loss, 'g', linewidth=2)
    plt.legend(['test', 'train'])
    plt.show(block=False)

# initializing RF with the number of trees, equal to the number of iterations
# with minimum loss on GB with learning_rate=0.2
forest = RandomForestClassifier(n_estimators=37,
                                random_state=241)
forest.fit(X_train,
           y_train)

# observing losses on RF
print(f'''
for RandomForestClassifier with 37 trees
test: {log_loss(y_test, forest.predict_proba(X_test))}
''')
plt.show()
