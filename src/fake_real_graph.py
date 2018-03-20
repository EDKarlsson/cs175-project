from sklearn.cluster import AgglomerativeClustering
import sklearn
from sklearn import manifold
import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import os

try:
    import preprocessing as pp
except:
    import src.preprocessing as pp

np.random.seed(0)

limit = 1000


def load_documemts_tfidvector(lim=28000):
    fake_articles = os.listdir(pp.DATA_DIR + "/generated_articles/")
    data_set = []
    for fa in fake_articles:
        if fa != '.DS_Store':
            f = open(pp.DATA_DIR + "/generated_articles/" + fa)
            data_set.append(f.readline())
            f.close()
    fake_articles = data_set
    data_set = list(set(pp.load_data()['content']))
    merged = data_set[:lim] + fake_articles[:lim]

    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(merged)


time0 = time.time()

vectors = load_documemts_tfidvector(limit)
vector_labels = ["real", "fake"]

print("Took : %.2fs " % (time.time() - time0))

vectors = vectors.toarray()

X = vectors[:, 1:]


# ----------------------------------------------------------------------
# Visualize the clustering
def plot_clustering(X_red, labels, title=None):
    x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
    X_red = (X_red - x_min) / (x_max - x_min)

    plt.figure(figsize=(6, 4))
    # for i in range(X_red.shape[0]):
    r = plt.scatter(X_red[:limit, 0], X_red[:limit, 1], c='r', s=5)
    f = plt.scatter(X_red[limit:, 0], X_red[limit:, 1], c='b', s=5)
    plt.legend((r,f), ('Real', 'Fake'))
    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title, size=17)
    plt.axis('off')
    plt.tight_layout()


print("Computing embedding")
X_red = manifold.SpectralEmbedding(n_components=2).fit_transform(X)
print("Done. : %.2fs " % (time.time() - time0))

t0 = time.time()

print("Plotting Graph")
plot_clustering(X_red, [0] * (limit) + [1] * (limit), "Real Vs. Fake : {} Data Points".format(limit*2))
print("Done : %.2fs " % (time.time() - time0))

plt.show()
