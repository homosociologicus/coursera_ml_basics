import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score


# column -1 is the target, others are features
data = pd.read_csv('abalone.csv')

# encoding Sex column
data.Sex = data.Sex.map(
    lambda x: 1 if x == 'M' else (
        -1 if x == 'F' else 0
    )
)

# preparing for CV
k_fold = KFold(shuffle=True, random_state=1)
quality = []

# getting rounded r2 scores for RF with different number of trees
for k in range(1, 51):
    quality.append(
        round(
            np.mean(
                cross_val_score(
                    RandomForestRegressor(
                        n_estimators=k,
                        random_state=1
                    ),
                    data.iloc[:, :-1],
                    data.iloc[:, -1],
                    scoring='r2',
                    cv=k_fold
                )
            ),
            2
        )
    )

# observing which value of k is optimal for > .52 score
print(f'''minimum number of trees for > 0.52 score:
{(np.array(quality) > .52).argmax() + 1}''')
