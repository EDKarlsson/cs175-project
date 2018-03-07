from sklearn.cluster import AgglomerativeClustering
from sklearn import manifold
import numpy as np
from matplotlib import pyplot as plt

try:
    import preprocessing as pp
except:
    import src.preprocessing as pp
import time

np.random.seed(0)

words = 500

print("Starting : %.2fs " % (time.time()))

vectors, vector_labels = pp.get_vectors()
vectors = vectors.toarray()

# np.random.shuffle(vectors)  # Shuffle the data points
print(vector_labels)
X = vectors[:words, 1:]


# ----------------------------------------------------------------------
# Visualize the clustering
def plot_clustering(X_red, X, labels, title=None):
    x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
    X_red = (X_red - x_min) / (x_max - x_min)

    plt.figure(figsize=(6, 4))
    for i in range(X_red.shape[0]):
        plt.text(X_red[i, 0], X_red[i, 1], str(vector_labels[labels[i]]),
                 color=plt.cm.spectral(labels[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title, size=17)
    plt.axis('off')
    plt.tight_layout()


print("Computing embedding : %.2fs " % (time.time()))
X_red = manifold.SpectralEmbedding(n_components=50).fit_transform(X)
print("Done.")

print("Starting Clustering".format(time.time()))
# linkage = ["ward", "complete", "average"]
link = "ward"
# for link in linkage:
#     print("Linkage : {}".format(link))
clustering = AgglomerativeClustering(n_clusters=8)
t0 = time.time()
clustering.fit(X_red)
# print("%s : %.2fs" % (link, time.time() - t0))

print("Plotting cluster : %.2fs " % (time.time()))
plot_clustering(X_red, X, clustering.labels_, "%s linkage" % link)

plt.show()
