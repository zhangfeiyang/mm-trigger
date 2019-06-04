#!/usr/bin/env python
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate, Lambda
from tensorflow.keras import optimizers

import create_easy_model
import create_general_model
import gen_dataset
import gen_dataset2

#model = create_easy_model.create_model()
model = create_general_model.create_model()
import os
if not os.path.exists('Data'):
    datas = gen_dataset2.gen_data()
    data = datas[0]
    label = datas[1]

else:
    file = open('Data','r')
    datas = []
    labels = []
    lines = file.readlines()
    N = len(lines)
    #for line in file:
    for i in range(N):
        line = lines[i]
        ds = line.split()
        if len(ds)>1:
            data = []
            for d in ds:
                data.append(float(d))
            datas.append(data)
        else:
            labels.append([int(ds[0])])
    import numpy as np
    data = np.array(datas)
    label = np.array(labels,'i')


#model.compile(optimizer=tf.train.AdamOptimizer(0.01),
#                      #loss=tf.nn.sigmoid_cross_entropy_with_logits,       # mean squared error
#                      #loss='sparse_categorical_crossentropy',       # mean squared error
#                      loss='binary_crossentropy',       # mean squared error
#                      #loss='mse',       # mean squared error
#                      metrics=['accuracy'])
#                      #metrics=['mae'])
#                      #metrics=tf.metrics.mean_per_class_accuracy)
#                      #metrics=tf.metrics.mean_per_class_accuracy)
##model.compile(optimizer=tf.train.AdamOptimizer(),
##                      loss=tf.nn.sigmoid_cross_entropy_with_logits)

#model.compile('sgd', loss=tf.keras.losses.BinaryCrossentropy())
#model.compile('sgd', loss='binary_crossentropy')
#model.compile('adam', loss='binary_crossentropy')
model.compile(optimizer=tf.train.AdamOptimizer(),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
model.fit(data,label,epochs=10,batch_size=10)

tests = []
test_label = []
for i in range(int(N/2.0)):
    tests.append(data[i])
    test_label.append(label[i])
tests = np.array(tests)
result = model.predict(tests)

#for i in range(int(N/2.0)):
#    print(result[i],test_label[i][0])

