import keras
import src.preprocessing as preproc
import numpy as np

article_embedding = preproc.load_article_embedding()
article_embedding = np.array(article_embedding)

Xtr = article_embedding[:-1]
Ytr = article_embedding[1:]

model = keras.Sequential()
# Each add is a layer
model.add(keras.layers.Embedding(128, 128))  # Embedding layer
model.add(keras.layers.LSTM(1024))
model.add(keras.layers.Dense(512, activation='sigmoid'))
model.add(keras.layers.Dense(128, activation='sigmoid'))
model.add(keras.layers.Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()
