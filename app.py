import gradio as gr

import tensorflow as tf
import keras
from matplotlib import pyplot as plt
import numpy as np

objects =  tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = objects.load_data()

training_images  = training_images / 255.0
test_images = test_images / 255.0

from keras.layers import Flatten, Dense
model = tf.keras.models.Sequential([Flatten(input_shape=(28,28)), 
                                    Dense(256, activation='relu'),
                                    Dense(256, activation='relu'),
                                    Dense(128, activation='relu'), 
                                    Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=10)

test=test_images[0].reshape(-1,28,28)
pred=model.predict(test)
print(pred)

def predict_image(img):
  img_3d=img.reshape(-1,28,28)
  im_resize=img_3d/255.0
  prediction=model.predict(im_resize)
  pred=np.argmax(prediction)
  return pred

iface = gr.Interface(predict_image, inputs="sketchpad", outputs="label")

iface.launch(debug='True')