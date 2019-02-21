import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
# from tensorflow.python.keras import Sequential, initializers, optimizers
# from tensorflow.python.keras.layers import Flatten, Dense, Dropout
mnist = input_data.read_data_sets('MNIST_Fashion/', one_hot=True)
from tensorflow import keras
import tensorflow as tf

num_inputs = 784
num_outputs = 10
num_hiddens1 = num_hiddens2 = 256
lr = 0.5
num_epochs = 10

model = keras.Sequential()
model.add(keras.layers.Dense(num_hiddens1, kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                bias_initializer='zeros', activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(num_hiddens2, kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                bias_initializer='zeros', activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(num_outputs, kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                bias_initializer='zeros', activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(lr), metrics=['acc'])

for epoch in range(num_epochs):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size=256)
    history = model.fit(batch_xs, batch_ys, shuffle=True)
    print('epoch', epoch + 1, ', loss=', history.history['loss'])

loss, accuracy = model.evaluate(mnist.test.images, mnist.test.labels)
print('loss=', loss)
print('accuracy=', accuracy * 100, '%')


a = np.argmax(model.predict(mnist.test.images), axis=1)
b = np.argmax(mnist.test.labels, axis=1)
print(np.mean(np.equal(a, b)))
