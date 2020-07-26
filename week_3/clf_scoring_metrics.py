import pandas as pd

from numpy import where
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import roc_auc_score, precision_recall_curve

# first dataset
# columns: true and predicted classes
data = pd.read_csv('classification.csv', index_col=None, dtype=bool)

# creating confusion matrix
correct = data.pred == data.true
confusion_matrix = pd.DataFrame(
    [[data[correct & data.true].shape[0], data[~correct & ~data.true].shape[0]],
     [data[~correct & data.true].shape[0], data[correct & ~data.true].shape[0]]],
    index=['Predicted Positive', 'Predicted Negative'],
    columns=['Actual Positive', 'Actual Negative'],
    dtype=int
)

# observing different metrics
print(
    'confusion matrix:',
    confusion_matrix,
    f'accuracy: {accuracy_score(data.true, data.pred)}',
    f'precision: {precision_score(data.true, data.pred)}',
    f'recall: {recall_score(data.true, data.pred)}',
    f'f1: {f1_score(data.true, data.pred)}',
    sep='\n\n',
    end='\n\n'
)


# second dataset
# columns: true classes and estimations by different classifiers
scores = pd.read_csv('scores.csv')

for column in scores.columns[1:]:

    # finding a classifier with maximum precision provided recall >= 70%
    prec_roc = precision_recall_curve(scores.true, scores[column])
    recall_more_70 = prec_roc[0][where(prec_roc[1] >= 0.7)]

    # also observing aur-roc metric
    print(f'max precision for {column} is {max(recall_more_70)}',
          f'auc-roc for {column}: {roc_auc_score(scores.true, scores[column])}',
          sep='\n',
          end='\n\n')
