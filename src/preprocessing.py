import pandas as pd
import os
import nltk
import pickle
from keras.utils import to_categorical as tc
import keras.preprocessing.text as ktext
import numpy as np
from sklearn.preprocessing import OneHotEncoder as OHEncode
import unicodedata

global CRAP_CHAR
CRAP_CHAR = 0

global NUM_VOCAB
NUM_VOCAB = 1000


def load_data():
    DATA_DIR = "../data"
    try:
        ARTICLES = os.listdir(DATA_DIR)
    except:
        print("../data directory not found")
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


def make_string(lim=150000, types='all'):
    if types == 'all':
        return load_data()['content'][:lim]
    else:
        data = load_data()
        data = data[data['publication'] == types]
        return data['content'][:lim]


def pair_word_punctuation(string_list:list):
    pairs = ""
    for s in string_list:
        tokens = nltk.word_tokenize(s)
        for i, token in enumerate(tokens):
            if token in ['.', ';', ':', '!', '-', '@', '#', '$', '%', '^', '&']:
                pairs += " " + tokens[i-1] + token
        s += pairs


def make_sequences(lim=150000, types='all', format='word'):
    global NUM_VOCAB
    if format == 'word':
        tokenizer = ktext.Tokenizer(num_words=NUM_VOCAB - 1, filters='')
        string = make_string(lim, types)
        pair_word_punctuation(string)
        # print(string[0])
        tokenizer.fit_on_texts(string)
        encoded_text = tokenizer.texts_to_sequences(string)
        print(len(encoded_text[0]))

        # create -> word sequences
        sequences = list()
        print('reaching')
        for l in encoded_text:
            for i in range(1, len(l)):
                sequence = l[i - 1:i + 1]
                sequences.append(sequence)
    else:
        data = make_string(lim, types)
        chars = set()
        for d in data:
            for c in d:
                chars.add(c)
        unique = np.unique(list(chars))

        char_map = {c: i for i, c in enumerate(unique)}
        print(char_map)

        sequences = list()
        for d in data:
            for i in range(1, len(d)):
                sequence = [char_map[c] for c in d[i - 1:i + 1]]
                sequences.append(sequence)

        unique_values = np.unique(sequences)
        NUM_VOCAB = len(unique_values)
    sequences = np.array(sequences)

    x_train, y_train = sequences[:, 0], sequences[:, 1]

    if format == 'word':
        reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
    else:
        tokenizer = char_map
        reverse_word_map = dict(map(reversed, tokenizer.items()))
    return x_train, y_train, tokenizer, reverse_word_map


if __name__ == '__main__':
    print('derp')
    load_data()
    make_sequences(1)

    # load_article_embedding()

    # data = load_data()
    # # fdist = bag_of_words(data['content'][0])
    # fdist = nltk.FreqDist()
    # for i, article in enumerate(data['content']):
    #     fdist += bag_of_words(article)
    #     if i % 1000 == 0:
    #         print(i)
    # pickle.dump(fdist, open('../../out/all_articles.txt', mode = 'wb'))
