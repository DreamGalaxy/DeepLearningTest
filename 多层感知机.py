import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_Fashion/", one_hot=True)
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# 初始化模型参数
num_inputs = 784
num_outputs = 10
num_hiddens1 = 256
num_hiddens2 = 256

batch_size = 256
num_epochs = 1000
lr = 0.002

X = tf.placeholder(tf.float32, [None, num_inputs])
y_true = tf.placeholder(tf.float32, [None, num_outputs])

W1 = tf.Variable(tf.random_normal(stddev=0.01, shape=(num_inputs, num_hiddens1)))
b1 = tf.Variable(tf.zeros(num_hiddens1))
W2 = tf.Variable(tf.random_normal(stddev=0.01, shape=(num_hiddens1, num_hiddens2)))
b2 = tf.Variable(tf.zeros(num_hiddens2))
W3 = tf.Variable(tf.random_normal(stddev=0.01, shape=(num_hiddens2, num_outputs)))
b3 = tf.Variable(tf.zeros(num_outputs))


H1 = tf.nn.relu(tf.matmul(X, W1)+b1)
H2 = tf.nn.relu(tf.matmul(H1, W2)+b2)
y_pred = tf.nn.softmax(tf.matmul(H2, W3)+b3)

# 自定义交叉熵损失函数
# cross_entropy = -tf.reduce_sum(y_true*tf.log(y_pred))
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_true*tf.log(y_pred), axis=1))

# 交叉熵损失函数
cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=tf.matmul(H2, W3) + b3))

# 准确率
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1)), "float"))

train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for epoch in range(1, num_epochs + 1):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={X: batch_xs, y_true: batch_ys})

print('准确率:', sess.run(accuracy, feed_dict={X: mnist.test.images, y_true: mnist.test.labels}) * 100, '%')


# # 显示图片函数
# def show_fashion_imgs(images):
#     _, figs = plt.subplots(1, len(images), figsize=(15, 15))
#     for f, img in zip(figs, images):
#         f.imshow(img.reshape((28, 28)))
#         f.axes.get_xaxis().set_visible(False)
#         f.axes.get_yaxis().set_visible(False)
#     plt.show()
#
#
# # 输出正确lables函数
# def get_text_labels(labels):
#     # text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
#     #                'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
#     text_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#     l = np.argmax(labels, 1)
#     res = []
#     for i in l:
#         res.append(text_labels[i])
#     return res
#
# # 使用模型测试,选取的测试图片可以变
# sample = mnist.test.images[100:109]
# label = mnist.test.labels[100:109]
# print('real labels:', get_text_labels(label))
# SAM = tf.placeholder(tf.float32, [None, 784])
# LAB = tf.placeholder(tf.float32, [None, 10])
# pred_label = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(SAM, W1)+b1), W2) + b2)
# print('predictions:', get_text_labels(sess.run(pred_label, feed_dict={SAM: sample, y_true: label})))
# show_fashion_imgs(sample)
