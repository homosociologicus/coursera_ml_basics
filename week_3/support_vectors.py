import pandas as pd

from sklearn.svm import SVC


data = pd.read_csv('svm-data.csv', header=None)

# choosing C=1e5 because of linear separability of data
clf = SVC(C=100000, kernel='linear', random_state=241)
clf.fit(data.iloc[:, 1:], data.iloc[:, 0])

print('support vectors:', *clf.support_)
