import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_Fashion/", one_hot=True)
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 显示图片函数
def show_fashion_imgs(images):
    _, figs = plt.subplots(1, len(images), figsize=(15, 15))
    for f, img in zip(figs, images):
        f.imshow(img.reshape((28, 28)))
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


# 输出正确lables函数
def get_text_labels(labels):
    # text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
    #                'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    text_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    l = np.argmax(labels, 1)
    res = []
    for i in l:
        res.append(text_labels[i])
    return res

# 初始化模型参数
num_inputs = 784
num_outputs = 10
W = tf.Variable(tf.random_normal(shape=(num_inputs, num_outputs), stddev=0.01))
b = tf.Variable(tf.zeros(num_outputs))

num_batchs = 1000
lr = 0.01
batch_size = 100

X = tf.placeholder(tf.float32, [None, num_inputs])
y_true = tf.placeholder(tf.float32, [None, num_outputs])


# 定义Softmax回归模型
def softmax(o):
    exp = tf.exp(o)
    partition = tf.reduce_sum(exp, axis=1, keepdims=True)
    return exp / partition  # 此处用到了广播机制


def net(X):
    return softmax(tf.matmul(X, W)+b)

# y_pred = net(X)

# tensorflow定义Softmax模型，两个y_pred选一个即可
y_pred = tf.nn.softmax(tf.matmul(X, W) + b)

# 自定义交叉熵损失函数
# cross_entropy = -tf.reduce_sum(y_true*tf.log(y_pred))

# tensorFlow提供的封装好的交叉熵损失函数，相比自定义有优化
cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=tf.matmul(X, W) + b))


# 计算准确率
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(1, num_batchs + 1):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={X: batch_xs, y_true: batch_ys})

    print('准确率:', sess.run(accuracy, feed_dict={X: mnist.test.images, y_true: mnist.test.labels})*100, '%')

    # 使用模型测试,选取的测试图片可以变
    sample = mnist.test.images[233:243]
    label = mnist.test.labels[233:243]
    SAM = tf.placeholder(tf.float32, [None, num_inputs])
    LAB = tf.placeholder(tf.float32, [None, num_outputs])
    pred_label = net(SAM)
    probability = sess.run(pred_label, feed_dict={SAM: sample, y_true: label})
    print(probability)
    print('predictions:', get_text_labels(probability))
    print('real labels:', get_text_labels(label))
    show_fashion_imgs(sample)
