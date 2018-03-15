try:
    from src.models import kmodel
except:
    import kmodel

try:
    import src.preprocessing as pp
except:
    import preprocessing as pp

import nltk
import numpy as np
import sklearn

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

        if not seed:
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


def get_fake_article_set(model, data, tokenizer, lim=10):
    print('Generating %d articles' % lim)
    fake_articles = []
    real_articles = data['content'][:lim].tolist()
    for i, art in enumerate(real_articles):
        print('Generating article #%d' % i)
        fake_articles.append(gen_fake_article(model, tokenizer, art))

    X = real_articles + fake_articles
    Y = np.zeros(2 * lim)
    Y[lim:] = 1
    return X, Y


def get_train_data(X, Y):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y)
    print('Finished generating articles')

    return x_train, x_test, y_train, y_test


def create_bag_of_words(article):
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

    X, Y = get_fake_article_set(model, data, kmodel.tokenizer)

    print("Creating bags")
    bags = [create_bag_of_words(a) for a in X]

    x_train, x_test, y_train, y_test = get_train_data(bags, Y)

    print('Starting Training')
    lr = sklearn.linear_model.LogisticRegression(n_jobs=4)
    lr.fit(x_train, y_train)
    score = lr.score(x_test, y_test)
    print('Score', score)
