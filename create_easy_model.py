#!/usr/bin/env python
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
  
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate, Lambda
from tensorflow.keras import optimizers

BECs = [115,  113,    111,    112,    108,    115,    115,    108,    112,    111,    113,    119,    123,    126,    124,    124,    125,  124,  123,    125,    124,    124,    126,    125,    115,    116,    117,    114,    116,    115,    116,    117,    114,    117,  116,  118,    129,    130,    132,    129,    131,    127,    130,    131,    129,    132,    130,    129,    120,    121,    120,  118,  121,    120,    121,    121,    118,    120,    121,    121,    128,    131,    131,    128,    131,    128,    129,    131,  128,  131,    131,    129,    128,    131,    131,    128,    131,    129,    128,    131,    128,    131,    131,    129,    122,  123,  123,    121,    123,    122,    123,    123,    121,    123,    123,    123,    131,    130,    134,    131,    134,    130,  131,  134,    131,    134,    132,    133,    117,    119,    120,    117,    120,    117,    118,    120,    117,    120,    119,  119,  126,    128,    126,    126,    127,    126,    125,    127,    126,    126,    128,    126,    119,    117,    115,    115,  113,  118,    119,    113,    115,    115,    117,    119,    115,    113,    111,    112,    108,    115,    115,    108,    112,  111,  113,    119,    123,    126,    124,    124,    125,    124,    123,    125,    124,    124,    126,    125,    115,    116,  117,  114,    116,    115,    116,    117,    114,    117,    116,    118,    129,    130,    132,    129,    131,    127,    130,  131,  129,    132,    130,    129,    120,    121,    120,    118,    121,    120,    121,    121,    118,    120,    121,    121,  128]

def first_layer(inputTensor):
        
    groups = []         # One GCU is one group
    index = 0 
    for i in range(144):            # there are 144 BECs, so ~42 GCUs for one BEC
        group = Lambda(lambda x: x[:,index:index+BECs[i]*4],  output_shape=((4,)))(inputTensor)
        index += BECs[i]*4
        print(index)
        group = Dense(1)(group)
        groups.append(group)

    return groups

def second_layer(last_layer):

    groups = []
    for i in range(7):            # there are 7 RMUs, so ~21 BECs for one RMU
        group = Concatenate()(last_layer[i:i+21])
        group = Dense(1)(group)
        groups.append(group)

    return groups

def third_layer(last_layer):
    
    group = Concatenate()(last_layer)
    return group


def create_model():
        
    inputTensor = Input((17739*4,))        # there are 6174 GCUs, and 3 PMTs for every GCU, (x,y,z,t) input for every PMT
    First = first_layer(inputTensor)
    Second=second_layer(First)
    Third = third_layer(Second)

    outputTensor = Dense(1)(Third)
    model = Model(inputTensor,outputTensor)

    return model


if __name__ == "__main__":
        
    model = create_model()
    model.summary()  
