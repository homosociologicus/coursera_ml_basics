import pandas as pd

from numpy import where
from sklearn.tree import DecisionTreeClassifier


data = pd.read_csv('titanic.csv', index_col='PassengerId')

# recoding Sex column
data['SexInt'] = where(data['Sex'] == 'male', 0, 1)

# extracting train data
train = data[['Pclass', 'Fare', 'Age', 'SexInt', 'Survived']].dropna()
X_train = train.iloc[:, :-1]
y = train.iloc[:, -1]

# training the model
clf = DecisionTreeClassifier(random_state=241)
clf.fit(X_train, y)

# extracting the answer
feat_import = clf.feature_importances_

print('Feature importances:',
      *zip(X_train.columns, feat_import),
      sep='\n')
