import tensorrt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import activations
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import pandas as pd

import gradio as gr
from gradio import Interface
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import numpy as np

import pandas as pd
import matplotlib.image as mpimg


X_train = pd.read_csv("csvTrainImages 60k x 784.csv")
y_train = pd.read_csv("csvTrainLabel 60k x 1.csv")
X_test = pd.read_csv("csvTestImages 10k x 784.csv")
y_test = pd.read_csv("csvTestLabel 10k x 1.csv")

print('X_train shape', X_train.shape)
print('y_train shape', y_train.shape)
print('X_test shape', X_test.shape)
print('y_test shape', y_test.shape)

X_train /= 255
X_test /= 255

X_train = X_train.values.reshape(-1,28,28,1)
X_test = X_test.values.reshape(-1,28,28,1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


print("Training matrix shape", X_train.shape)
print("Testing matrix shape", X_test.shape)

nb_classes = 10

Y_train = to_categorical(y_train, nb_classes)
Y_test = to_categorical(y_test, nb_classes)

print(nb_classes)
print(Y_train.shape)
print(Y_test.shape)

import sklearn
from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=4)

print(X_train.shape)
print(Y_train.shape)
print(X_val.shape)
print(Y_val.shape)

#Creating CNN model

model = Sequential()

model.add(layers.Conv2D(filters = 80, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))

model.add(layers.MaxPool2D(pool_size=(2,2)))

model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', activation ='relu'))

model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Dropout(0.25))


model.add(layers.Flatten())

model.add(layers.Dense(128, activation = "relu"))

model.add(layers.Dropout(0.25))

model.add(layers.Dense(10, activation = "softmax"))

model.summary()

from tensorflow.keras.optimizers import SGD
optimizer = SGD(learning_rate=0.001, momentum=0.30)

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

history = model.fit( X_train,Y_train, batch_size=64, epochs = 1, validation_data = (X_val, Y_val), verbose = 1)

def predict_image(img):
  img_3d = img.reshape(-1, 28,28)
  img_scaled = img_3d/255
  prediction = model.predict(img_scaled)
  pred = np.argmax(prediction)

  return pred.item()


iface = gr.Interface(predict_image, inputs='sketchpad', outputs='label', title='Arabic Numbers Recognition', description='Draw a number')

iface.launch()