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

inputTensor = Input((12,))

group1 = Lambda(lambda x: x[:,:3],  output_shape=((3,)))(inputTensor)
group2 = Lambda(lambda x: x[:,3:6], output_shape=((3,)))(inputTensor)
group3 = Lambda(lambda x: x[:,6:9], output_shape=((3,)))(inputTensor)
group4 = Lambda(lambda x: x[:,9:],  output_shape=((3,)))(inputTensor)

group1 = Dense(1)(group1) 
group2 = Dense(1)(group2)
group3 = Dense(1)(group3)
group4 = Dense(1)(group4)

sec_group1 = Concatenate()([group1,group2])
sec_group2 = Concatenate()([group3,group4])

sec_group1 = Dense(1)(sec_group1)
sec_group2 = Dense(1)(sec_group2)

third_group = Concatenate()([sec_group1,sec_group2])

outputTensor = Dense(1)(third_group)

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
    for j in range(12):
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


model.fit(datas,labels,epochs=10,batch_size=1000)
test = []

for i in range(12):
    test.append(i)
test = [test]
test = np.array(test)
result = model.predict(test)
print(result)
model.save_weights('./checkpoints/my_checkpoint')
#model.fit(datas,labels,epochs=5)
model.summary()

reader = tf.train.NewCheckpointReader('./checkpoints/my_checkpoint')
all_variables = reader.get_variable_to_shape_map()

weights = {}
bias = {}
for key in all_variables:
        if 'layer_with_weights' in key and 'VARIABLE_VALUE' in key and not 'optimizer' in key:
            #print("tensor_name: ", key)
            #print(reader.get_tensor(key))
            keys = key.split('/')
            index = int(keys[0][-1])
            if keys[1] == 'kernel':
                weights[index] = reader.get_tensor(key) 
            elif keys[1] == 'bias':
                bias[index] = reader.get_tensor(key) 

Sum = [0,0,0,0]
for i in range(12):
    Sum[int(i/3)] += (weights[int(i/3)][i%3]*test[0][i])

for i in range(4):
    Sum[i] += bias[i]
    #print(Sum[i])

A = Sum[0]*weights[4][0] + Sum[1]*weights[4][1] + bias[4]
B = Sum[2]*weights[5][0] + Sum[3]*weights[5][1] + bias[5]

print(A*weights[6][0] + B*weights[6][1] + bias[6])

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
