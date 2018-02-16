import pandas as pd
from scipy.sparse import csr_matrix
import os
import nltk
import pickle

def load_data():
    DATA_DIR = "../../data"
    ARTICLES = os.listdir(DATA_DIR)
    return pd.concat([pd.read_csv(DATA_DIR + "/" + f) for f in ARTICLES if '.csv' in f], ignore_index=True)

def bag_of_words(article):
    tokens = nltk.word_tokenize(article)
    fdist = nltk.FreqDist(tokens)
    remove_stopwords(fdist)
    return fdist


def remove_stopwords(tokens, stopwords=set(nltk.corpus.stopwords.words('english'))):
    for key in list(tokens.keys()):
        if key in stopwords:
            del tokens[key]



if __name__ == '__main__':
    data = load_data()
    # fdist = bag_of_words(data['content'][0])
    fdist = nltk.FreqDist()
    for i, article in enumerate(data['content']):
        fdist += bag_of_words(article)
        if i % 1000 == 0:
            print(i)
    pickle.dump(fdist, open('../../out/all_articles.txt', mode = 'wb'))
