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
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

x_train, y_train, tokenizer, word_map = preproc.make_sequences(100000, types='Fox News')

model_type = 'Fox' # string to define which folder to store trained models in
h1_size = 100
epochs = 100000


def get_latest_model(model_type=model_type):
    try:
        name = os.listdir('saved_models/' + model_type)
        iteration = sorted(name)[-1]
        return int(iteration), keras.models.load_model('saved_models/' + model_type + '/' + iteration)
    except:
        return -1, None

def save_model(model, iteration):
    model.save('saved_models/' + model_type + '/' + str(iteration))

def define_model():
    name, model = get_latest_model()
    if model != None:
        return model

    model = keras.models.Sequential()
    # Each add is a layer
    model.add(keras.layers.Embedding(preproc.NUM_VOCAB, h1_size, input_length=1))  # Embedding layer
    model.add(keras.layers.LSTM(1024))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(preproc.NUM_VOCAB, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    model.summary()
    return model

def sample_output(model, word_seed, num_samples=100):
    seed = np.array(tokenizer.texts_to_sequences([word_seed]))

    for i in range(num_samples):
        word_prob = model.predict(seed)[0]

        num = np.random.random()
        for j, p in enumerate(word_prob):
            num -= p

            if num < 0:
                seed = np.array([j + 1])
                break

        print(' ' + word_map[seed[0]], end='')

    print()


def train_model(model, epochs=epochs):
    model_iter, _ = get_latest_model()
    print('Initial model num:', model_iter)
    for e in range(epochs):
        model.fit(x_train, y_train, verbose=1, epochs=1)

        # save model
        model_iter += 1
        save_model(model, model_iter)

        # print some sample output
        print('Epoch:', model_iter)
        sample_output(model, 'the')




if __name__ == '__main__':
    print('vocab size:', preproc.NUM_VOCAB)
    model = define_model()
    train_model(model)
