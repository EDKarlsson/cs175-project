import pandas as pd
import os
import nltk
import keras.preprocessing.text as ktext
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
import numpy as np
import string as pystring
import pickle
# import textrank
from sklearn.feature_extraction.text import TfidfVectorizer

CURRENT_DIR = os.getcwd()
DATA_DIR = ""
if "src" in CURRENT_DIR:
    DATA_DIR = CURRENT_DIR.replace("src", "data")
elif CURRENT_DIR.split('/')[-1] == "cs175-project":
    DATA_DIR = CURRENT_DIR + "/data"

global CRAP_CHAR
CRAP_CHAR = 0

global NUM_VOCAB
NUM_VOCAB = 3000


def create_corpus():
    """
    Method will load the entire article data set then convert into a PlainTextCorpusReader. It will remove stop words
    and punctuations from the articles. Writes it to a file then returns the NLTK PlainTextCorpusReader
    :return: PlainTextCorpusReader
    """
    print("Creating Corpus")
    data = make_string()
    corpus_filename = "corpus"
    s = ""
    stopwords = set(nltk.corpus.stopwords.words('english'))
    for d in data:
        # d = d.lower()
        for sw in stopwords:
            d = d.replace(" " + sw + " ", " ")
        for p in pystring.punctuation:
            d = d.replace(p + " ", " ").replace("\t", " ").replace(" " + p, " ").replace("“", "").replace("”", "")
        s += (d + "\n\n")

    print("Writing corpus to file.")
    f = open('{}.txt'.format(corpus_filename), 'w')
    f.write(s)
    f.close()
    newcorpus = PlaintextCorpusReader('.', '{}.txt'.format(corpus_filename))
    return newcorpus


def load_data():
    """
    Loads the News article data set and returns a Pandas Dataframe containing all news articles
    :return: Pandas:DataFrame
    """
    print("Loading Data")
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


def create_summaries(beg=0, end=100, save_per_iter=0):
    """
    Creates summaries using TextRank. This is currently not used.
    """
    print("Generating Summaries...")
    data = load_data().as_matrix()[beg:end]

    article_sum = []
    for i, article in enumerate(data):
        print('articles/{}'.format(beg + i))
        article_sum.append({
            'publisher': article[3],
            'title': article[2],
            'keyphrases': textrank.extract_key_phrases(article[9]),
            'summary': textrank.extract_sentences(article[9], summary_length=200)
        })
        if save_per_iter > 0 and i % save_per_iter == 0 and i > 0:
            pickle.dump(article_sum,
                        open("summerized_articles_{}-{}.dat".format(i - save_per_iter + 1 + beg, i + 1 + beg), "wb"))

    if save_per_iter == 0:
        pickle.dump(article_sum, open("summerized_articles_{}-{}.dat".format(beg, end), "wb"))


def load_summerized_data():
    """
    Method will load the summarized data set
    :return: Summarized articles
    """
    print("Loading Summerized Data")
    return pickle.load(open("summerized_articles_0-100.dat", "rb"))


def get_summerized_articles():
    """
    Returns just the summaries in the summarized article data set.
    :return: Set of String Summaries
    """
    print("Retrieving Summerized Articles")
    art = load_summerized_data()
    return [a['summary'] for a in art]


def load_sentence_tokens(limit=10000, publication=None, overwrite=False):
    """
    Will try to load previously saved tokenized sentences. If it does not exist then it will load the news article
    dataset and tokenize each article into sentences and pickle it to a tokens file.
    :param limit: Number of sentences to retrieve
    :param publication: Only create sentences for specific publications (This will not work if already created)
    :param overwrite: Overwrite the previously created sentence tokens
    :return: Sentences
    """
    SENTENCE_TOKENS_FILE = "sentence.tokens"
    print("Loading sentence tokens")
    if overwrite or SENTENCE_TOKENS_FILE not in set(os.listdir(DATA_DIR)):
        data = load_data()
        sentences = []
        print("Tokenizing Sentences")
        if publication:
            data = data[data['publication'] == publication]
        for article in data['content']:
            sentences.extend(nltk.sent_tokenize(article))
        if SENTENCE_TOKENS_FILE not in set(os.listdir(DATA_DIR)):
            pickle.dump(sentences, open(DATA_DIR + "/" + SENTENCE_TOKENS_FILE, "wb"))
    else:
        sentences = pickle.load(open(DATA_DIR + "/" + SENTENCE_TOKENS_FILE, "rb"))
    return sentences[:limit]


def get_vectors(type="", remake_binary=False):
    """
    Creates TFID vectors for the data set if it does not already exists. If it does, load the vectors and return them.
    :param remake_binary: Recreate the TFID Vector files
    :return: TFID Vectors, Publication Labels
    """
    print("Loading vectors...")
    if "src" in CURRENT_DIR:
        os.chdir("..")
        data_dir = os.getcwd() + "/data"
        os.chdir(CURRENT_DIR)
    else:
        data_dir = CURRENT_DIR + "/data"

    vectorizer = TfidfVectorizer()
    publishers = []

    if type == "fake":
        fake_articles = os.listdir(data_dir+ "/generated_articles/")
        data_set = []
        for fa in fake_articles:
            if fa != '.DS_Store':
                f = open(data_dir + "/generated_articles/" + fa)
                data_set.append(f.readline())
        # data_set = list(set(data_set))
        vectors = vectorizer.fit_transform(data_set)
    else:
        if remake_binary or os.path.isfile(data_dir + "tfid_vectors") == False:
            data = load_data()
            publishers = list(set(data['publication']))
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
    """
    Creates a bag of words from the article then creating a frequency distribution, removing stop words then returning it.
    There is the option to remove punctuations from the articles if the parameter is provided. By default this is false.
    :param article: Article String
    :param remove_punc: Boolean
    :return: FreqDist
    """
    print("Creating bag of words")
    if remove_punc:
        for p in pystring.punctuation:
            article = article.replace(p + " ", " ").replace("\t", " ").replace(" " + p, " ").replace("“", "").replace(
                "”", "").replace('’ ', ' ').replace(". ", " ")
    tokens = nltk.word_tokenize(article)
    fdist = nltk.FreqDist(tokens)
    remove_stopwords(fdist)
    return fdist


def remove_stopwords(tokens, stopwords=set(nltk.corpus.stopwords.words('english'))):
    """
    Removes stop words from the list of tokens based on the english stopwords in NLTK.
    :param tokens: List of tokens
    :param stopwords: NLTK Stopwords
    :return: List of Tokens
    """
    print("Removing stopwords")
    for key in list(tokens.keys()):
        if key in stopwords:
            del tokens[key]


def make_string(lim=150000, types='all'):
    """
    For each article in the entire data set, retrieve and return the specified number of articles.
    :param lim: Limit of articles
    :param types: Publications
    :return: List of Articles
    """
    print("Making strings")
    if types == 'all':
        return load_data()['content'][:lim]
    else:
        data = load_data()
        data = data[data['publication'] == types]
        return data['content'][:lim]


def insert_split_token(grams, token):
    """
    Takes a gram and a split token, then creates a sentence with the token located in between the grams.
    :param grams: NGram of words
    :param token: Token to insert
    :return: String
    """
    ms = ""
    for gram in grams:
        for word in gram:
            ms += (word + " ")
        ms += (token + " ")
    return ms


def make_ngram(sentence: str, n):
    """
    Splits a sentence into n-grams
    :param sentence: String of a sentence
    :param n: Number of grams to split to
    :return: Sentence -> ngram
    """
    return nltk.ngrams(sentence.split(), n)


def sep_word_punctuation(string_list: list):
    """
    Separates the words from punctuations in the sentences. e.g. Hello! -> Hello !
    :param string_list: List of strings
    :return: List of strings
    """
    print("Separating punctuations from words")
    unique = set(c for article in string_list for c in article)
    # unique = [c for c in unique if (not str(c).isalpha() and not str(c).isnumeric() and c != '\'')]
    unique = open("token_set.tokens", "r").readline()
    unique = eval(unique)
    for i, s in enumerate(string_list):
        sp = s
        for u in unique:
            sp = sp.replace(u, " " + u + " ")
        string_list[i] = sp
    return string_list


def make_sequences(lim=10000, types='all', format='word', split=" ", ngram=0, article_type="whole"):
    """
    Creates sequences for training, a tokenizer, a word map, sentence lengths and seeds to use.
    Method will either load sentences, articles or words to be turned into training data. Which ever is used will be
    given to a tokenizer that is used to create a word map. The word map is an index mapping to each word.
    :param lim: Number of articles or sentences to create
    :param types: Publication Name
    :param format: Word, Character
    :param split: Split token for tokenizer
    :param ngram: NGram to create <- Deprecated
    :param article_type: Summaries, Sentences, Whole articles
    :return: XTrain, YTrain, tokenizer, word map, length of sentences, seeds.
    """
    len_of_sentences = 0
    seeds = None
    global NUM_VOCAB
    print("Making Sequences")
    if article_type == "summary":
        articles = get_summerized_articles()
    elif article_type == "sentences":
        articles = load_sentence_tokens(limit=lim)
    else:
        articles = make_string(lim, types)
    if format == 'word':
        tokenizer = ktext.Tokenizer(num_words=NUM_VOCAB - 1, filters='”“"#$%&()*+,-/.!?:;<=>@[\\]^_`{|}~\t\n',
                                    split=split)
        # if article_type != "sentences":
        # articles = sep_word_punctuation(articles)

        # pair_word_punctuation(string)
        # print(string[0])
        print("Fitting tokenizer on text")
        tokenizer.fit_on_texts(articles)
        print("Texts to sequences")
        encoded_text = tokenizer.texts_to_sequences(articles)

        # always use sentences to sample
        sent_art = load_sentence_tokens(limit=lim)
        sent_tok = tokenizer.texts_to_sequences(sent_art)
        len_of_sentences = [len(a) for a in sent_tok]

        seeds = [a[0] for a in encoded_text if a]

        # create -> word sequences
        sequences = list()
        print('reaching')
        for l in encoded_text:
            for i in range(1, len(l)):
                sequence = l[i - 1:i + 1]
                sequences.append(sequence)
    else:
        chars = set()
        for d in articles:
            for c in d:
                chars.add(c)
        unique = np.unique(list(chars))

        print("Creating character map")
        char_map = {c: i for i, c in enumerate(unique)}
        print(char_map)

        sequences = list()
        for d in articles:
            for i in range(1, len(d)):
                sequence = [char_map[c] for c in d[i - 1:i + 1]]
                sequences.append(sequence)

        unique_values = np.unique(sequences)
        NUM_VOCAB = len(unique_values)
    sequences = np.array(sequences)

    x_train, y_train = sequences[:, 0], sequences[:, 1]

    print("Creating reverse word map")
    if format == 'word':
        word_map = dict(map(reversed, tokenizer.word_index.items()))
    else:
        tokenizer = char_map
        word_map = dict(map(reversed, tokenizer.items()))
    return x_train, y_train, tokenizer, word_map, len_of_sentences, seeds


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
