import numpy as np

from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_score, KFold


file = load_boston()
x = scale(file['data'])
y = file['target']
k_fold = KFold(shuffle=True, random_state=42)
quality = []

for p in np.linspace(1, 10, num=200):
    quality.append((
        np.mean(
            cross_val_score(
                KNeighborsRegressor(
                    n_neighbors=5,
                    weights='distance',
                    p=p
                ),
                x,
                y=y,
                scoring='neg_mean_squared_error',
                cv=k_fold
            )
        ),
        p
    ))

print(f'Least loss, corresponding Minkowski power parameter: {max(quality)}')
