import tensorflow as tf
from tensorflow import keras
import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.preprocessing import normalize

mnist = tf.keras.datasets.mnist
(trainx,trainy),(testx,testy)=mnist.load_data()

trainx=trainx.reshape(trainx.shape[0],trainx.shape[1],trainx.shape[2],1).astype('float32')
testx=testx.reshape(testx.shape[0],testx.shape[1],testx.shape[2],1).astype('float32')

trainy=tf.keras.utils.to_categorical(trainy,10)
testy=tf.keras.utils.to_categorical(testy,10)

filter_input=(5,5)
filter_hidden=(5,5)
data_format='channels_last'
pool=(2,2)

trainx/=255
testx/=255

model=keras.Sequential()

model=keras.Sequential()

model.add(keras.layers.Conv2D(20,filter_input,activation='relu',input_shape=(28,28,1)))
model.add(keras.layers.MaxPool2D(pool))
model.add(keras.layers.Conv2D(50,filter_hidden,activation='relu'))
model.add(keras.layers.MaxPool2D(pool))



model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(500,activation='relu'))
model.add(keras.layers.Dense(10,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(trainx,trainy,validation_data=(testx,testy),epochs=2,batch_size=32)
#Batch size

model.summary()
start=time.time()
model.predict(testx)
print(time.time()-start)
#test_loss, test_acc = model.evaluate(testx,testy)
#print('Test accuracy:', test_acc)