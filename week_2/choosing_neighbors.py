import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale


# analizing raw and scaled data, observing differences in quality
raw_quality = []
clean_quality = []
k_fold = KFold(shuffle=True, random_state=42)

data = pd.read_csv('wine.data', header=None)
X_raw = data.iloc[:, 1:]
X_clean = scale(data.iloc[:, 1:])
y = data.iloc[:, 0]

# filling mean accuracy on CV
for k in range(1, 51):
    raw_quality.append(
        np.mean(
            cross_val_score(
                KNeighborsClassifier(n_neighbors=k),
                X_raw,
                y=y,
                scoring='accuracy',
                cv=k_fold
            )
        )
    )
    clean_quality.append(
        np.mean(
            cross_val_score(
                KNeighborsClassifier(n_neighbors=k),
                X_clean,
                y=y,
                scoring='accuracy',
                cv=k_fold
            )
        )
    )

print(f'''
Max quality with raw data:
{max(raw_quality)} with {raw_quality.index(max(raw_quality)) + 1} neighbors.

Max quality with scaled data:
{max(clean_quality)} with {clean_quality.index(max(clean_quality)) + 1} neighbors.'''
)
