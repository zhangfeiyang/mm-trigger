#!/usr/bin/env python
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
  
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate, Lambda
from tensorflow.keras import optimizers


def create_model():

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(17739, )),
        keras.layers.Dense(1000, activation=tf.nn.relu),
        keras.layers.Dense(2, activation=tf.nn.softmax)
    ])        
    
    return model


if __name__ == "__main__":
        
    model = create_model()
    model.summary()  
