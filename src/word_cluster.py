from sklearn.cluster import AgglomerativeClustering
from sklearn import manifold
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time


def load_vectors():
    return pd.read_csv('../data/corpus_vectors.txt', sep=" ", header=None)


words = 100

print("Loading vectors...")
vectors = load_vectors().as_matrix()
np.random.shuffle(vectors)  # Shuffle the data points
vector_labels = vectors[:words, 0]
print(vector_labels)
X = vectors[:words, 1:]

np.random.seed(0)


# ----------------------------------------------------------------------
# Visualize the clustering
def plot_clustering(X_red, X, labels, title=None):
    x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
    X_red = (X_red - x_min) / (x_max - x_min)

    plt.figure(figsize=(6, 4))
    for i in range(X_red.shape[0]):
        plt.text(X_red[i, 0], X_red[i, 1], str(vector_labels[i]),
                 color=plt.cm.spectral(labels[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title, size=17)
    plt.axis('off')
    plt.tight_layout()


print("Computing embedding")
X_red = manifold.SpectralEmbedding(n_components=50).fit_transform(X)
print("Done.")

print("Starting Clustering")
# linkage = ["ward", "complete", "average"]
link = "ward"
# for link in linkage:
#     print("Linkage : {}".format(link))
clustering = AgglomerativeClustering(n_clusters=8)
t0 = time.time()
clustering.fit(X_red)
# print("%s : %.2fs" % (link, time.time() - t0))

plot_clustering(X_red, X, clustering.labels_, "%s linkage" % link)

plt.show()
