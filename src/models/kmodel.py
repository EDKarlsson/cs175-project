import sys

sys.path.append('..')
import keras

try:
    import preprocessing as preproc
except:
    import src.preprocessing as preproc

import pandas as pd
import numpy as np
import os
import tensorflow as tf
import argparse
from keras.backend.tensorflow_backend import set_session
from keras.regularizers import l2

# Parse command line and setting configurations
parser = argparse.ArgumentParser()
parser.add_argument("--publication", type=str, help="Set the model type [publication]", default="all")
parser.add_argument("--epochs", type=int, help="Number of epochs", default=30)
parser.add_argument("--vocab", type=int, help="Size of the vocabulary.", default=5000)
parser.add_argument("--articles", type=int, help="Number of articles", default=10000)
parser.add_argument("--ngram", type=int, help="Use NGram to split strings", default=0)
parser.add_argument("--verbose", type=bool, help="Verbose Keras output", default=True)
parser.add_argument("--saveperepoch", type=int, help="Save model every x-epoch", default=1)
parser.add_argument("--lstm", type=int, help="Units per LSTM layer in RNN", default=512)
parser.add_argument("--gpu_memory", type=float, help="Set GPU Memory Limit", default=.8)
parser.add_argument("--split", type=str, help="Char or string to split each sentence on.", default=" ")
parser.add_argument("--art_type", type=str, help="Type of article tokens. Sentences, summaries, whole", default="")
args = parser.parse_args()

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = args.gpu_memory
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

format = 'word'
model_type = args.publication  # string to define which folder to store trained models in
preproc.NUM_VOCAB = args.vocab

x_train, y_train, tokenizer, word_map, len_of_sentences, seeds_list = preproc.make_sequences(args.articles,
                                                                                             types=model_type,
                                                                                             format=format,
                                                                                             ngram=args.ngram,
                                                                                             split=args.split,
                                                                                             article_type=args.art_type)

h1_size = 50
epochs = args.epochs



def load_vectors():
    return pd.read_csv('../data/corpus_vectors.txt', sep=" ", header=None)


def create_vector_embedding(in_dim=preproc.NUM_VOCAB, out_dim=h1_size, input_length=1):
    '''
    Creates a custom vector embedding layer for keras.
    :param in_dim:
    :param out_dim:
    :param input_length:
    :return:
    '''
    word_vectors = {w[0]: w[1:] for w in load_vectors().as_matrix()}
    # assemble the embedding_weights in one numpy array
    n_symbols = len(word_map) + 1  # adding 1 to account for 0th index (for masking)
    embedding_weights = np.zeros((in_dim, 50))
    for index, word in word_map.items():
        if index < in_dim:
            embedding_weights[index, :] = word_vectors[word] if word in word_vectors.keys() else np.random.random(50)

    print(np.count_nonzero(embedding_weights == 0, axis = 0))
    print(embedding_weights)
    # define inputs here
    embedding_layer = keras.layers.Embedding(output_dim=out_dim, input_dim=in_dim, trainable=True,
                                             input_length=input_length)
    embedding_layer.build((None,))  # if you don't do this, the next step won't work
    embedding_layer.set_weights([embedding_weights])

    return embedding_layer


def get_latest_model(model_type=model_type):
    try:
        name = os.listdir('saved_models/' + model_type)
        iteration = sorted(name)[-1]
        return int(iteration), keras.models.load_model('saved_models/' + model_type + '/' + iteration)
    except:
        return -1, None


def save_model(model, iteration):
    model.save('saved_models/' + model_type + '/' + str(iteration))


def create_model(units=args.lstm, dropout=.2):
    model = keras.models.Sequential()
    corpus_vectors = load_vectors().as_matrix()
    # Each add is a layer
    # model.add(keras.layers.Embedding(preproc.NUM_VOCAB, h1_size, input_length=1))  # Embedding layer
    model.add(create_vector_embedding())
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(units, recurrent_dropout=dropout)))
    model.add(keras.layers.Dense(preproc.NUM_VOCAB, activation='softmax', kernel_initializer='he_normal'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    model.summary()
    return model


def define_model():
    name, model = get_latest_model()
    if model != None:
        return model
    return create_model()


def sample_output(model, word_seed, tokenizer, num_samples=5, temperature=1.):
    if format == 'word':
        seed = np.array(tokenizer.texts_to_sequences([word_seed]))
    else:
        seed = np.array([tokenizer[word_seed]])

    sentence = ""
    print("SEED ", seed)
    sentence += word_map[seed[0][0]]
    for i in range(num_samples):
        print("Sentence #", i)
        for _ in range(np.random.choice(len_of_sentences)):
            word_prob = model.predict(seed)[0]

            word_prob = np.log(word_prob) / temperature
            word_prob = np.exp(word_prob) / np.sum(np.exp(word_prob))
            seed = np.random.choice(range(word_prob.shape[0]), p=word_prob)
            seed = np.array([seed])

            if format == 'word':
                sentence += ' ' + word_map[seed[0]]
            else:
                sentence += word_map[seed[0]]
        sentence = sentence[0].upper() + sentence[1:]
        print("{}. ".format(sentence))
        seed = np.array([np.random.choice(seeds_list)])
        sentence = word_map[seed[0]]
    print()


def train_model(model, epochs=epochs):
    model_iter, _ = get_latest_model()
    print('Initial model num:', model_iter)

    filepath = 'saved_models/'
    if os.path.isdir('saved_models') is False:
        os.mkdir('saved_models')

    filepath += model_type

    if os.path.isdir(filepath) is False:
        os.mkdir(filepath)

    filepath += '/{epoch}'

    class saver(keras.callbacks.ModelCheckpoint):
        def on_epoch_end(self, epoch, logs=None):
            if ((epoch - 1) % args.saveperepoch == 0):
                super().on_epoch_end(epoch, logs)

    class sampler(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            pass
            sample_output(self.model, 'the', tokenizer=tokenizer)

    model.fit(x_train, y_train, verbose=args.verbose, epochs=model_iter + epochs + 1, batch_size=16,
              initial_epoch=model_iter + 1,
              callbacks=[sampler(), saver(filepath=filepath, verbose=1)])
    return model


if __name__ == '__main__':
    print('vocab size:', preproc.NUM_VOCAB)
    model = define_model()
    train_model(model, args.epochs)
