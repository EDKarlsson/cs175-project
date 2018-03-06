import sys

sys.path.append('..')
import keras

try:
    import preprocessing as preproc
except:
    import src.preprocessing as preproc

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
parser.add_argument("--vocab", type=int, help="Size of the vocabulary.", default=3000)
parser.add_argument("--articles", type=int, help="Number of articles", default=600)
parser.add_argument("--ngram", type=int, help="Use NGram to split strings", default=0)
parser.add_argument("--verbose", type=bool, help="Verbose Keras output", default=True)
parser.add_argument("--saveperepoch", type=int, help="Save model every x-epoch", default=1)
parser.add_argument("--lstm", type=int, help="Units per LSTM layer in RNN", default=512)
args = parser.parse_args()

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))

format = 'word'
model_type = args.publication # string to define which folder to store trained models in

x_train, y_train, tokenizer, word_map = preproc.make_sequences(args.articles, types=model_type, format=format,
                                                               ngram=args.ngram)

h1_size = 50
epochs = args.epochs

preproc.NUM_VOCAB = args.vocab


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
    # Each add is a layer
    model.add(keras.layers.Embedding(preproc.NUM_VOCAB, h1_size, input_length=1))  # Embedding layer
    model.add(keras.layers.LSTM(units, recurrent_dropout=dropout))
    model.add(keras.layers.Dense(preproc.NUM_VOCAB, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    model.summary()
    return model


def define_model():
    name, model = get_latest_model()
    if model != None:
        return model
    return create_model()


def sample_output(model, word_seed, num_samples=100):
    if format == 'word':
        seed = np.array(tokenizer.texts_to_sequences([word_seed]))
    else:
        seed = np.array([tokenizer[word_seed]])

    print(word_seed, end='')
    for i in range(num_samples):
        word_prob = model.predict(seed)[0]

        num = np.random.random()
        for j, p in enumerate(word_prob):
            num -= p

            if num <= 0:
                seed = np.array([j])
                break
        if format == 'word':
            print(' ' + word_map[seed[0]], end='')
        else:
            print(word_map[seed[0]], end='')
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
                super().on_epoch_end(epochs, logs)

    class sampler(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            sample_output(self.model, 'the', num_samples=100)

    model.fit(x_train, y_train, verbose=args.verbose, epochs=model_iter + epochs + 1, initial_epoch=model_iter + 1,
              callbacks=[saver(filepath=filepath, verbose=1), sampler()])
    return model


if __name__ == '__main__':
    print('vocab size:', preproc.NUM_VOCAB)
    model = define_model()
    train_model(model)
