try:
    from src.models import kmodel
except:
    import kmodel

try:
    import src.preprocessing as pp
except:
    import preprocessing as pp

import pickle
import nltk
import numpy as np
import sklearn
import os

SEED = 2

CURRENT_DIR = os.getcwd()

DATA_DIR = CURRENT_DIR.split('cs175-project')[0] + "cs175-project/data"

stopwords = set(nltk.corpus.stopwords.words('english'))
vocab = {w: i for i, w in kmodel.word_map.items() if i <= pp.NUM_VOCAB}


def load():
    """
    Loads the Keras LSTM RNN and loads the news article data set and returns them.
    :return: LSTM, Article Data Set
    """
    return kmodel.define_model(), pp.load_data()


def gen_fake_article(model, tokenizer, article, temperature=1.):
    """
    Takes a trained model, a tokenizer that has a sequence of word tokens and a template article then synthesizes a fake
    article based on seed words from the template article. The sentence length and article length will be identical but
    words will be different.
    :param model: Keras LSTM RNN
    :param tokenizer: Keras Text Tokenizer
    :param article: Template article for seeding
    :param temperature: Increase sampling variance
    :return: returns a fake article
    """
    parsed_sentences = nltk.sent_tokenize(article)

    fake_article = ""
    for sent in parsed_sentences:
        words = sent.split()
        seed = np.array(tokenizer.texts_to_sequences([words[0]]))

        # print(seed)
        # print(words)
        if seed.shape != (1, 1):
            continue
        start_word = words[0]
        sentence = start_word[0].upper() + start_word[1:]
        for _ in range(len(words) - 1):
            word_prob = model.predict(seed)[0]

            word_prob = np.log(word_prob) / temperature
            word_prob = np.exp(word_prob) / np.sum(np.exp(word_prob))
            seed = np.random.choice(range(word_prob.shape[0]), p=word_prob)
            seed = np.array([seed])

            sentence += ' ' + kmodel.word_map[seed[0]]
        model.predict(seed)
        fake_article += sentence + ". "
    return fake_article


def get_fake_article_set(model, data, tokenizer, lim=10, overwrite=False, ):
    """
    This method will generate a set of fake articles based on the model and data set provided.
    :param model: Keras LSTM RNN
    :param data: Pandas data set of articles
    :param tokenizer: Keras Text Tokenizer
    :param overwrite: Overwrite the already saved fake articles
    :param lim: Limit the number of articles to generate
    :return:
    """
    print('Generating %d articles' % lim)
    fake_articles = []
    real_articles = data['content'][:lim].tolist()
    generated_data_dir = DATA_DIR + "/generated_articles"
    generated_articles = set(os.listdir(generated_data_dir))

    for i, art in enumerate(real_articles):
        if not overwrite and 'a{}.txt'.format(i) in generated_articles:
            print('Loading article #%d' % i)
            f = open(generated_data_dir + '/a{}.txt'.format(i), 'r')
            fake_articles.append(f.readline())
            f.close()
        else:
            print('Generating article #%d' % i)
            fake_article = gen_fake_article(model, tokenizer, art)
            fake_articles.append(fake_article)
            f = open(generated_data_dir + '/a{}.txt'.format(i), 'w')
            f.write(fake_article)
            f.close()

    # check for bad simulations
    print("Creating bag of words for articles")
    fake_articles = np.array([create_bag_of_words(a) for a in fake_articles])
    real_articles = np.array([create_bag_of_words(a) for a in real_articles])

    # Remove articles that could be generate since the seed was not in the vocab
    good_sims = np.any(fake_articles, axis=1)
    print('Found', np.count_nonzero(good_sims), 'Good articles')

    fake_articles = fake_articles[good_sims]
    real_articles = real_articles[good_sims]

    fake_train, fake_test, ftrain_target, ftest_target = get_train_data(fake_articles, np.ones(len(fake_articles)))
    real_train, real_test, rtrain_target, rtest_target = get_train_data(real_articles, np.zeros(len(real_articles)))

    # X = real_articles + fake_articles
    # Y = np.zeros(2 * lim)
    # Y[lim:] = 1

    return np.concatenate([fake_train, real_train]), np.concatenate([fake_test, real_test]), np.concatenate(
        [ftrain_target, rtrain_target]), \
           np.concatenate([ftest_target, rtest_target])


def get_train_data(X, Y):
    """
    Given X and Y data, shuffle the data and create a X,Y Training set and X,Y Testing set.
    :param X: Data points
    :param Y: Labels
    :return: Xtrain, Ytrain, Xtest, Ytest
    """
    print("Shuffling Data")
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, random_state=SEED)
    return x_train, x_test, y_train, y_test


def create_bag_of_words(article):
    """
    Create a bag of words for the articles
    :param article:
    :return:
    """
    non_stopwords = [i for w, i in vocab.items() if w not in stopwords]

    bag_of_words = np.zeros(pp.NUM_VOCAB + 1)

    tokens = kmodel.tokenizer.texts_to_sequences([article])[0]

    tokens = list(set(tokens))

    bag_of_words[tokens] = 1
    bag_of_words = bag_of_words[non_stopwords]
    bag_of_words = bag_of_words[1:]

    return bag_of_words


if __name__ == "__main__":
    model, data = load()

    x_train, x_test, y_train, y_test = get_fake_article_set(model, data, kmodel.tokenizer, 1000)

    pickle.dump(x_test[:200], open(DATA_DIR + "/x_test_sample.txt", "wb"))
    pickle.dump(y_test[:200], open(DATA_DIR + "/y_test_sample.txt", "wb"))
    print(x_train.shape)
    print('Starting Training')
    lr = sklearn.linear_model.LogisticRegression(n_jobs=4)
    lr.fit(x_train, y_train)
    pickle.dump(lr, open(DATA_DIR + "/logistic_regression.model", "wb"))
    score = lr.score(x_test, y_test)
    print('Score', score)
