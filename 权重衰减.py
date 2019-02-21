import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import set_matplotlib_formats


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

lr = 0.03
num_epochs = 10

X = tf.placeholder(tf.float32, [None, num_inputs])
y = tf.placeholder(tf.float32, [None, 1])

W = tf.Variable(tf.random_normal(shape=(num_inputs, 1), stddev=1))
b = tf.Variable(tf.zeros(shape=(1,)))
y_pred = tf.matmul(X, W) + b

loss = tf.losses.mean_squared_error(labels=y, predictions=y_pred)

lambd = 0
train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss + lambd * tf.reduce_sum(tf.pow(W, 2)) / 2)


def set_figsize(figsize=(3.5, 2.5)):
    set_matplotlib_formats('retina')  # 打印高清图。
    plt.rcParams['figure.figsize'] = figsize  # 设置图的尺寸。


# 画图函数
def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals)
        plt.legend(legend)
    plt.show()

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    train_ls = []
    test_ls = []
    for epoch in range(1, num_epochs+1):
        train_l, train_ = sess.run([loss, train_step], feed_dict={X: train_features, y: train_labels})
        print('epoch', epoch, 'loss=', train_l)
        train_ls.append(train_l)
        test_l = sess.run(loss, feed_dict={X: test_features, y: test_labels})
        test_ls.append(test_l)

    semilogy(range(1, num_epochs+1), train_ls, 'epochs', 'loss',
             range(1, num_epochs+1), test_ls, ['train', 'test'])

    print('W[0:10]=', sess.run(W[0:10]))

