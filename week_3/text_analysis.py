import numpy as np

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import KFold, GridSearchCV


# loading and processing text data
newsgroups = fetch_20newsgroups(subset='all',
                                categories=['alt.atheism', 'sci.space'])
tfidf = TfidfVectorizer()
X_vector = tfidf.fit_transform(newsgroups.data)
y = newsgroups.target

# preparing grid search on CV
grid = {'C' : np.power(10.0, np.arange(-5, 6))}
cv = KFold(shuffle=True,
           random_state=241)
clf = SVC(kernel='linear',
          random_state=241)
gs = GridSearchCV(clf,
                  grid,
                  scoring='accuracy',
                  cv=cv)

# looking for best regularization parameter C
gs.fit(X_vector,
       y)
print(f'best parameters for SVC: {gs.best_params_}')

# training model with the best parameters
svm = gs.best_estimator_
svm.fit(X_vector,
        y)

# looking for 10 words with the highest absolute weights
# using agrpartition instead of argsort due to performance
# not concerning with the order as the final one is alphabetical
coefs = np.abs(svm.coef_.toarray().flatten())
indices = np.argpartition(coefs, -10)[-10:]
words = [tfidf.get_feature_names()[index] for index in indices]

print('10 words with the highest weights:', *sorted(words))
