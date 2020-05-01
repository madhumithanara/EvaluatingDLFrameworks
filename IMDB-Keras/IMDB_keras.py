import numpy
import tensorflow as tf
from tensorflow import keras
#from keras.datasets import imdb
from matplotlib import pyplot
# load the dataset
# (X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data()
# X = numpy.concatenate((X_train, X_test), axis=0)
# y = numpy.concatenate((y_train, y_test), axis=0)

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

model = keras.Sequential()
model.add(keras.layers.Embedding(top_words, 100, input_length=max_words))
#model.add(keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
#model.add(keras.layers.MaxPooling1D(pool_size=2))
model.add(keras.layers.LSTM(100))
model.add(keras.layers.Flatten())
#model.add(keras.layers.Dense(250, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=50,verbose=1)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))