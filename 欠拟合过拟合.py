import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import set_matplotlib_formats

lr = 0.03
num_epochs = 100
n_train = 100
n_test = 100
true_w = [1.2, -3.4, 5.6]
true_b = 5

features = np.random.normal(size=(n_train + n_test, 1))
poly_features = np.concatenate((features, np.power(features, 2), np.power(features, 3)), axis=1)
labels = np.sum(true_w * poly_features, axis=1, keepdims=True)+true_b
labels += np.random.normal(scale=0.1, size=labels.shape)

X = tf.placeholder(tf.float32, [n_train, 1])
y_true = tf.placeholder(tf.float32, [n_train, 1])

# 三阶多项式
# W = tf.Variable(tf.random_normal(shape=(3, 1), stddev=0.01))
# b = tf.Variable(tf.zeros(shape=(1,)))
# poly_X = tf.concat(axis=1, values=[X, tf.pow(X, 2), tf.pow(X, 3)])
# y_pred = tf.matmul(poly_X, W) + b

# 线性函数(欠拟合)
W = tf.Variable(tf.random_normal(shape=(1, 1), stddev=0.01))
b = tf.Variable(tf.zeros(shape=(1,)))
y_pred = tf.matmul(X, W) + b


loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


def set_figsize(figsize=(3.5, 2.5)):
    set_matplotlib_formats('retina')  # 打印高清图。
    plt.rcParams['figure.figsize'] = figsize  # 设置图的尺寸。


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

train_ls = []
test_ls = []
for epoch in range(1, num_epochs+1):
    train_l, train_ = sess.run([loss, train_step], feed_dict={X: features[0:n_train], y_true: labels[0:n_train]})

    # 过拟合
    # train_l, train_ = sess.run([loss, train_step], feed_dict={X: np.tile(features[0:2], (50, 1)), y_true: np.tile(labels[0:2], (50, 1))})

    print('epoch', epoch, 'loss=', train_l)
    train_ls.append(train_l)
    test_l = sess.run(loss, feed_dict={X: features[n_train:n_train+n_test], y_true: labels[n_train:n_train+n_test]})
    test_ls.append(test_l)

print(sess.run(W))
print(sess.run(b))

semilogy(range(1, num_epochs+1), train_ls, 'epochs', 'loss',
             range(1, num_epochs+1), test_ls, ['train', 'test'])