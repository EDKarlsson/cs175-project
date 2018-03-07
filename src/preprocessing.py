import pandas as pd
import os
import nltk
import keras.preprocessing.text as ktext
import numpy as np
import string as pystring
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

CURRENT_DIR = os.getcwd()

global CRAP_CHAR
CRAP_CHAR = 0

global NUM_VOCAB
NUM_VOCAB = 1000


def create_corpus():
    data = make_string()
    s = ""
    stopwords = set(nltk.corpus.stopwords.words('english'))
    for d in data:
        for sw in stopwords:
            d = d.replace(" " + sw + " ", " ")
        for p in pystring.punctuation:
            d = d.replace(p + " ", " ").replace("\t", " ").replace(" " + p, " ").replace("“", "").replace("”", "")
        s += d

    f = open('corpus.txt', 'w')
    f.write(s)


def load_data():
    DATA_DIR = "../data"
    print("Current directory " + os.getcwd())
    root_content = os.listdir(".")
    if "data" in root_content:
        ARTICLES = os.listdir("data")
    else:
        try:
            ARTICLES = os.listdir(DATA_DIR)
        except:
            print("../data directory not found")
            DATA_DIR = "../../data"
            ARTICLES = os.listdir(DATA_DIR)

    return pd.concat([pd.read_csv(DATA_DIR + "/" + f) for f in ARTICLES if '.csv' in f], ignore_index=True)


def get_vectors(remake_binary=False):
    print("Loading vectors...")
    if "src" in CURRENT_DIR:
        os.chdir("..")
        data_dir = os.getcwd() + "/data/"
        os.chdir(CURRENT_DIR)
    else:
        data_dir = CURRENT_DIR + "/data"

    if remake_binary or os.path.isfile(data_dir + "tfid_vectors") == False:
        data = load_data()
        publishers = list(set(data['publication']))
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(data['content'])
        print("Saving Vectors")
        pickle.dump(vectors, open(data_dir + "tfid_vectors", "wb"))
        pickle.dump(publishers, open(data_dir + "publishers", "wb"))
    else:
        print("Loading pickle vector data")
        vectors = pickle.load(open(data_dir + "tfid_vectors", "rb"))
        publishers = list(pickle.load(open(data_dir + "publishers", "rb")))
    return vectors, publishers


def bag_of_words(article, remove_punc=False):
    if remove_punc:
        for p in pystring.punctuation:
            article = article.replace(p + " ", " ").replace("\t", " ").replace(" " + p, " ").replace("“", "").replace(
                "”", "").replace('’ ', ' ').replace(". ", " ")
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


def insert_split_token(grams, token):
    ms = ""
    for gram in grams:
        for word in gram:
            ms += (word + " ")
        ms += (token + " ")
    return ms


def make_ngram(sentence: str, n):
    return nltk.ngrams(sentence.split(), n)


def sep_word_punctuation(string_list: list):
    unique = set(c for article in string_list for c in article)
    unique = [c for c in unique if (not str(c).isalpha() and not str(c).isnumeric() and c != '\'')]
    for i, s in enumerate(string_list):
        sp = s
        for u in unique:
            sp = sp.replace(u, " " + u + " ")

        string_list[i] = sp
    return string_list


def make_sequences(lim=150000, types='all', format='word', split=" ", ngram=0):
    global NUM_VOCAB
    if format == 'word':
        tokenizer = ktext.Tokenizer(num_words=NUM_VOCAB - 1, filters='', split=split)
        articles = make_string(lim, types)
        articles = sep_word_punctuation(articles)

        # pair_word_punctuation(string)
        # print(string[0])
        tokenizer.fit_on_texts(articles)
        encoded_text = tokenizer.texts_to_sequences(articles)

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
