import numpy as np

from pandas import DataFrame
from sklearn.cluster import KMeans
from skimage.io import imread
from skimage import img_as_float


def psnr(arr1, arr2, max_val=1):
    """Returns peak signal-to-noise ratio."""
    return 10 * np.log10(max_val ** 2 / np.mean(np.square(arr1 - arr2)))


# loading the image and transforming it from 3D to 2D
data = img_as_float(imread('parrots.jpg')).reshape(-1, 3)
scores = DataFrame(index=range(8, 21), columns=['mean', 'median'])

# looking for the minimum number of clusters (eyeballing from 8 to 20) to have
# PSNR > 20; filling clusters with either their mean or median
for n_clusters in range(8, 21):

    # training the model
    k_means = KMeans(n_clusters=n_clusters,
                     random_state=241)
    k_means.fit(data)

    # filling each cluster with mean and median
    labels = k_means.labels_
    data_mean = np.empty(data.shape)
    data_medn = np.empty(data.shape)
    for clust in range(n_clusters):
        clust_inds = np.where(labels == clust)
        data_mean[clust_inds] = np.mean(data[clust_inds], axis=0)
        data_medn[clust_inds] = np.median(data[clust_inds], axis=0)

    # calculating PSNR for each way of filling data, fixing n_clusters
    scores.loc[n_clusters, 'mean'] = psnr(data_mean, data)
    scores.loc[n_clusters, 'median'] = psnr(data_medn, data)

# observing answers
print('Minimum clusters for PSNR > 20:',
      (scores>20).idxmax(),
      sep='\n')
