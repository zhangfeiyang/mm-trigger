#!/usr/bin/env python3
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
  
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate, Lambda
from tensorflow.keras import optimizers

def first_layer(inputTensor):
        
    groups = []         # One GCU is one group
    for i in range(6174):
        group = Lambda(lambda x: x[:,i:i+12],  output_shape=((12,)))(inputTensor)
        group = Dense(1)(group)
        groups.append(group)

    return groups

def second_layer(last_layer):

    groups = []
    for i in range(144):            # there are 144 BECs, so ~42 GCUs for one BEC
        group = Concatenate()(last_layer[i:i+42])
        group = Dense(1)(group)
        groups.append(group)

    return groups

def third_layer(last_layer):
    
    groups = []
    for i in range(7):            # there are 7 RMUs, so ~21 BECs for one RMU
        group = Concatenate()(last_layer[i:i+21])
        group = Dense(1)(group)
        groups.append(group)

    return groups

def forth_layer(last_layer):
    
    group = Concatenate()(last_layer)
    return group

def create_model():
        
    inputTensor = Input((6174*3*4,))        # there are 6174 GCUs, and 3 PMTs for every GCU, (x,y,z,t) input for every PMT
    First = first_layer(inputTensor)
    Second=second_layer(First)
    Third = third_layer(Second)
    Forth = forth_layer(Third)

    outputTensor = Dense(1)(Forth)
    model = Model(inputTensor,outputTensor)

    return model


if __name__ == "__main__":
        
    model = create_model()
    model.summary()  
