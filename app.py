pip install tensorrt
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
