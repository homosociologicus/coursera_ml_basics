import pandas as pd

from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


train = pd.read_csv('perceptron-train.csv', header=None)
test = pd.read_csv('perceptron-test.csv', header=None)
x_train = train.iloc[:, 1:]
y_train = train.iloc[:, 0]
x_test = test.iloc[:, 1:]
y_test = test.iloc[:, 0]
clf = Perceptron(max_iter=5,
                 tol=None,
                 random_state=241)

# dealing with raw data
clf.fit(x_train, y_train)
accuracy = accuracy_score(y_test,
                          clf.predict(x_test))

# scaling data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

clf.fit(x_train_scaled, y_train)
accuracy_scaled = accuracy_score(y_test,
                                 clf.predict(x_test_scaled))

print(f'Difference in accuracy after scaling: {accuracy_scaled - accuracy}')
