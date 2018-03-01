import pandas as pd
import os
import nltk
import pickle

global CRAP_CHAR
CRAP_CHAR = 0


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


def save_char_dict():
    data = load_data()
    articles = data['content'][:200]

    st = set()
    asciiset = set()
    for a in articles:
        st.update({l for l in a if ord(l) < 128})
        asciiset.update({ord(l) for l in a if ord(l) < 128})

    print(st)
    st = sorted(st)
    d = {c: st.index(c) for c in st}
    print(d)
    pickle.dump(d, open('../../out/character_dict', mode='wb'))


def bin_matrix():
    data = load_data()
    article_embedding = []
    for a in data['content']:
        article_embedding += [ord(c) for c in a if ord(c) < 128]

    # print(mat)
    pickle.dump(article_embedding, open('../../out/article_embedding_long', mode='wb'))

    # return article_embedding


def load_article_embedding(type="long"):
    return pickle.load(open('../../data/article_embedding_{}'.format(type), mode='rb'))


if __name__ == '__main__':
    print('derp')

    bin_matrix()
    # load_article_embedding()

    # data = load_data()
    # # fdist = bag_of_words(data['content'][0])
    # fdist = nltk.FreqDist()
    # for i, article in enumerate(data['content']):
    #     fdist += bag_of_words(article)
    #     if i % 1000 == 0:
    #         print(i)
    # pickle.dump(fdist, open('../../out/all_articles.txt', mode = 'wb'))
