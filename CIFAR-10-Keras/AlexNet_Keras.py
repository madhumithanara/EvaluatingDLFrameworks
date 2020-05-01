import tensorflow as tf
from tensorflow import keras
import _pickle as pickle
import os
import numpy as np


(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

class AlexNet:
    def __init__(self):
        self.model=keras.Sequential()
        self.model.add(keras.layers.Conv2D(64, (3,3),strides=(2,2),padding='same', activation='relu',input_shape = (32,32,3),name='block1_conv1'))
        self.model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))
        #self.model.add(keras.layers.BatchNormalization())
        
        self.model.add(keras.layers.Conv2D(192,(3,3),padding='same',activation='relu'))
        self.model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
        
        self.model.add(keras.layers.Conv2D(384,(3,3),padding='same',activation='relu'))
        self.model.add(keras.layers.Conv2D(256,(3,3),padding='same',activation='relu'))
        self.model.add(keras.layers.Conv2D(256,(3,3),padding='same',activation='relu'))
        self.model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))
        
        self.model.add(tf.layers.Flatten())

        self.model.add(keras.layers.Dense(4096,activation='relu'))
        self.model.add(keras.layers.Dense(4096,activation='relu'))
        self.model.add(keras.layers.Dense(10,activation='softmax'))
        self.model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model1=AlexNet()
model1.model.summary()
#x_train,y_train,x_test,y_test=load_cifar()
model1.model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=128,epochs=30,verbose=1)
#Batch size