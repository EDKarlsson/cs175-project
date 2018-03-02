import sys
sys.path.append('..')
import keras
import preprocessing as preproc
import numpy as np

x_train, y_train = preproc.make_sequences(1000)


h1_size = 100
epochs = 10


model = keras.models.Sequential()
# Each add is a layer
model.add(keras.layers.Embedding(preproc.NUM_VOCAB, h1_size, input_length=1))  # Embedding layer
model.add(keras.layers.LSTM(1024))
model.add(keras.layers.Dense(preproc.NUM_VOCAB, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
model.summary()

model.fit(x_train, y_train, verbose=1, epochs=epochs)
