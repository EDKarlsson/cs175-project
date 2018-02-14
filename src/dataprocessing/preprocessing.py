import pandas as pd
import numpy as np
import os
import nltk

def load_data():
    DATA_DIR ="../../data"
    ARTICLES = os.listdir(DATA_DIR)
    return pd.concat([pd.read_csv(DATA_DIR + "/" + f) for f in ARTICLES if '.csv' in f], ignore_index = True)


def bag_of_words(article):
    tokens = nltk.word_tokenize(article)
    tokens = remove_stopwords(tokens)
    fdist = nltk.FreqDist(tokens)
    return fdist
    
def remove_stopwords(tokens):
    return [word for word in tokens if word.isalpha() and word not in nltk.corpus.stopwords.words('english')]



if __name__ == '__main__':
    data = load_data()
    fdist = bag_of_words(data['content'][0])
