#!/usr/bin/env python
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate, Lambda
from tensorflow.keras import optimizers

inputTensor = Input((3,))

outputTensor = Dense(1)(inputTensor)

model = Model(inputTensor,outputTensor)

import numpy as np

import random
#data = np.random.random((100,12))
#labels = np.random.random((100,1))

datas = []
labels =[] 

for i in range(10000):
    data = []
    label = 0
    for j in range(3):
        x = (random.uniform(j,j+1))
        label += x
        data.append(x)
    #label /= 12
    datas.append(data)
    labels.append([label])


datas = np.array(datas)
labels = np.array(labels)
#print(datas)

#datas = np.random.random((100,12))
#labels = np.random.random((100,1))

#model.compile(
#            optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0),
#            #optimizer='rmsprop',
#              loss='sparse_categorical_crossentropy',
#              metrics=['accuracy'])

#model.compile(optimizer=tf.train.AdamOptimizer(0.001),
#              loss='categorical_crossentropy',
#              metrics=['accuracy'])

model.compile(optimizer=tf.train.AdamOptimizer(0.01),
                      loss='mse',       # mean squared error
                      metrics=['mae'])


model.fit(datas,labels,epochs=10,batch_size=100)
test = []

for i in range(3):
    test.append(i)
test = [test,test]
test = np.array(test)
result = model.predict(test)
print(result)
model.save_weights('./checkpoints/my_checkpoint')
#model.fit(datas,labels,epochs=5)
model.summary()

reader = tf.train.NewCheckpointReader('./checkpoints/my_checkpoint')
all_variables = reader.get_variable_to_shape_map()

for key in all_variables:
        if 'layer_with_weights' in key and 'VARIABLE_VALUE' in key and not 'optimizer' in key:
            print("tensor_name: ", key)
            print(reader.get_tensor(key))

#for tv in tf.trainable_variables():
#        print (tv.name)
#w = tf.get_default_graph().get_tensor_by_name("dense_1/kernel:0")
#
#with tf.Session() as sess:
#        tf.global_variables_initializer().run()
#        print(sess.run(w))
#print(tf.get_variable('v',[1])[0])

#for ele in all_variables:
#    print(ele)

#print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
#print(tf.all_variables())
#reader = tf.train.NewCheckpointReader('./checkpoints/my_checkpoint')
#all_variables = reader.get_variable_to_shape_map()
#print(all_variables)
