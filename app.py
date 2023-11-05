import gradio as gr
from gradio import Interface
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import numpy as np

(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()

X_train = np.concatenate((X_train, X_test))
y_train = np.concatenate((y_train, y_test))

X_train = X_train / 255
X_test = X_test / 255

data_augmentation = keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2, input_shape=(28, 28, 1)),
])

model = models.Sequential([   
    data_augmentation,
                                                   
    #cnn
    layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu'),
    layers.MaxPooling2D((2,2)),

    #dense

    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax'),

])

model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5)

def predict_image(img):
  img_3d = img.reshape(-1, 28,28)
  img_scaled = img_3d/255
  prediction = model.predict(img_scaled)
  pred = np.argmax(prediction)

  return pred.item()
    

iface = gr.Interface(predict_image, inputs='sketchpad', outputs='label', title='Digit Recognition Model By Debamrita Paul', description='Draw a single digit(0 to 9)', __gradio_theme='dark')

iface.launch()