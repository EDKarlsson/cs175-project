import sklearn.cluster
import numpy as np
import pandas as pd
import src.preprocessing as preproc


def load_vectors():
    return pd.read_csv('corpus_vectors.txt', sep=" ", header=None)


vectors = load_vectors()

target = vectors[:][0]
features = vectors[:][1:]

acluster = sklearn.cluster.AgglomerativeClustering(n_clusters=5)
