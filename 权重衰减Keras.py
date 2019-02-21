import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt

n_train = 20
n_test = 100
num_inputs = 200
true_w = np.ones((num_inputs, 1)) * 0.01
true_b = 0.05

features = np.random.normal(size=(n_train + n_test, num_inputs))
labels = np.dot(features, true_w) + true_b
labels += np.random.normal(scale=0.01, size=labels.shape)
train_features, test_features = features[:n_train, :], features[n_train:, :]
train_labels, test_labels = labels[:n_train], labels[n_train:]


# 画图函数
def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(5.5, 5.5)):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals)
        plt.legend(legend)
    plt.show()


lr = 0.03
num_epochs = 10
model = keras.Sequential()

lambd = 0
model.add(keras.layers.Dense(1, kernel_initializer='random_normal', kernel_regularizer=keras.regularizers.l2(lambd), bias_initializer='zeros'))

model.compile(optimizer=tf.keras.optimizers.SGD(lr),
              loss='mse',
              metrics=['acc'])

train_ls = []
test_ls = []
for epoch in range(num_epochs):
    history = model.fit(train_features, train_labels, shuffle=True, batch_size=10)
    print('epoch', epoch + 1, ', loss=', history.history['loss'])
    train_ls.append(history.history['loss'])
    loss = model.evaluate(test_features, test_labels, batch_size=10)[0]
    test_ls.append(loss)

semilogy(range(1, num_epochs+1), train_ls, 'epochs', 'loss',
         range(1, num_epochs+1), test_ls, ['train', 'test'])
