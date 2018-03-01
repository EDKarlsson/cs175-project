import pandas as pd
import os
import nltk
import pickle
from keras.utils import to_categorical as tc
import keras.preprocessing.text as ktext
import numpy as np

global CRAP_CHAR
CRAP_CHAR = 0

global NUM_VOCAB
NUM_VOCAB = 10000


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


def make_string(lim=150000):
    return load_data()['content'][:lim]


def make_sequences(lim=150000):
    tokenizer = ktext.Tokenizer(num_words=NUM_VOCAB - 1)
    string = make_string(lim)
    tokenizer.fit_on_texts(string)
    encoded_text = tokenizer.texts_to_sequences(string)

    # create -> word sequences
    sequences = list()
    for l in encoded_text:
        for i in range(1, len(l)):
            sequence = l[i - 1:i + 1]
            sequences.append(sequence)

    print(sequences[0:3])
    sequences = np.array(sequences)
    print(sequences.shape)


    x_train, y_train = sequences[:, 0], sequences[:, 1]
    y_train = tc(y_train, num_classes=NUM_VOCAB)
    return x_train, y_train


if __name__ == '__main__':
    print('derp')

    # load_article_embedding()

    # data = load_data()
    # # fdist = bag_of_words(data['content'][0])
    # fdist = nltk.FreqDist()
    # for i, article in enumerate(data['content']):
    #     fdist += bag_of_words(article)
    #     if i % 1000 == 0:
    #         print(i)
    # pickle.dump(fdist, open('../../out/all_articles.txt', mode = 'wb'))
