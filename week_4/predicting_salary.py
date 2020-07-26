import pandas as pd

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.feature_extraction import DictVectorizer


# loading data
train = pd.read_csv('salary-train.csv')
test = pd.read_csv('salary-test-mini.csv')

# substituting everything but letters and numbers with spaces
train.FullDescription = train.FullDescription.str.lower().replace(
    '[^a-zA-Z0-9]',
    ' ',
    regex=True
)
test.FullDescription = test.FullDescription.str.lower().replace(
    '[^a-zA-Z0-9]',
    ' ',
    regex=True
)

# vectorizing words which accur at least in 5 texts
tfidf = TfidfVectorizer(min_df=5)
x_vector_train = tfidf.fit_transform(train.FullDescription)
x_vector_test = tfidf.transform(test.FullDescription)

# filling missing data
train.LocationNormalized.fillna('nan', inplace=True)
train.ContractTime.fillna('nan', inplace=True)
test.LocationNormalized.fillna('nan', inplace=True)
test.ContractTime.fillna('nan', inplace=True)

# one-hot encoding categorical features
dict_vect = DictVectorizer()
categ_train = dict_vect.fit_transform(
    train[['LocationNormalized', 'ContractTime']].to_dict('records')
)
categ_test = dict_vect.transform(
    test[['LocationNormalized', 'ContractTime']].to_dict('records')
)

# merging sparse text and categorical matrices
x_train = hstack([x_vector_train, categ_train])
x_test = hstack([x_vector_test, categ_test])

# training ridge regression
ridge = Ridge(random_state=241)
ridge.fit(x_train,
          train.SalaryNormalized)

# observing model salary predictions
print(f'predictions for test data:{ridge.predict(x_test)}')
