import numpy
import tensorflow as tf
from tensorflow import keras
#from keras.datasets import imdb
from matplotlib import pyplot

top_words = 5000
(X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(num_words=top_words)
# pad dataset to a maximum review length in words
max_words = 1000
X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_words)
X_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_words)

top_words = 5000
(X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(num_words=top_words)
# pad dataset to a maximum review length in words
max_words = 1000
X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_words)
X_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_words)

embedding_size=32
model=keras.Sequential()
model.add(keras.layers.Embedding(5000, embedding_size, input_length=max_words))
model.add(keras.layers.LSTM(100,dropout=0.5))
model.add(keras.layers.Dense(1, activation='sigmoid'))
#print(model.summary())
model.summary()

model.compile(loss='binary_crossentropy', 
             optimizer='adam', 
             metrics=['accuracy'])

batch_size = 64
num_epochs = 3
X_valid, y_valid = X_train[:batch_size], y_train[:batch_size]
X_train2, y_train2 = X_train[batch_size:], y_train[batch_size:]
model.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=num_epochs)