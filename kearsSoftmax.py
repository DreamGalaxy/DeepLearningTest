import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

model = keras.Sequential()
model.add(keras.layers.Dense(10, activation='softmax',
                             kernel_initializer='random_normal',
                             bias_initializer='zeros'))

lr = 0.01
model.compile(optimizer=tf.train.GradientDescentOptimizer(lr),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

data = mnist.train.images
labels = mnist.train.labels
num_epochs = 100
batch_size = 100
model.fit(data, labels, epochs=num_epochs, batch_size=batch_size)

model.evaluate(mnist.test.images, mnist.test.labels)

sample = mnist.test.images[100:109]
sample_labels = mnist.test.labels[100:109]
print('real labels:', sample_labels.argmax(axis=1))
print('predictions:', model.predict(sample, batch_size=10).argmax(axis=1))
