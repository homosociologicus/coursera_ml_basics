import numpy as np
import pandas as pd

from sklearn.decomposition import PCA


def explain_components(pca, percentage=75):
    # a little barbaric, but simple way
    sum = 0
    count = 0
    for rate in pca.explained_variance_ratio_:
        count += 1
        sum += rate
        if sum >= 0.01 * percentage:
            break
    return count


# loading train and test data
train = pd.read_csv('close_prices.csv')
pca = PCA(n_components=10)
pca.fit_transform(train.iloc[:, 1:])

test = pd.read_csv('djia_index.csv')
dji = test.iloc[:, 1].values  # Dow Jones Industrial Average data

# extracting pca components
first_comp = pca.transform(train.iloc[:, 1:])[:, 0]
comps = pca.components_

# observing different inputs of data into the DJI
print(f'''
number of components for 90% of variance: {explain_components(pca, percentage=90)}
correlation between 1st pca component and DJI: {np.corrcoef(first_comp, dji)[0, 1]}
company with the biggest input in the first component: {train.columns[comps[0].argmax() + 1]}''')
