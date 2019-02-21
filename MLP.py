from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
# 获取数据集
data_set = input_data.read_data_sets('MNIST_Fashion/', one_hot=True)

# 定义模型参数
num_inputs = 784  # 每幅图片的像素个数
num_outputs = 10  # 分类个数
num_hidden_units = 256  # 隐藏单元个数为256
learning_rate = 0.5
# 各层参数均匀分布的范围
temp1 = tf.sqrt(6/(num_inputs+num_hidden_units))
temp2 = tf.sqrt(6/(num_outputs+num_hidden_units))
# Xavier随机初始化
W1 = tf.Variable(tf.random_uniform(shape=(num_inputs, num_hidden_units), minval=-temp1, maxval=temp1))  # 隐藏层的权重
b1 = tf.Variable(tf.zeros(num_hidden_units))  # 隐藏层的偏差
W2 = tf.Variable(tf.random_uniform(shape=(num_hidden_units, num_outputs), minval=-temp2, maxval=temp2))  # 输出层的权重
b2 = tf.Variable(tf.zeros(num_outputs))  # 输出层的偏差
# 定义占位符
X = tf.placeholder(tf.float32, [None, num_inputs])
y_ = tf.placeholder(tf.float32, [None, num_outputs])

# 构造模型
H = tf.nn.relu(tf.matmul(X, W1) + b1)
Y_pred = tf.nn.softmax(tf.matmul(H, W2) + b2)  # 隐藏层的输出作为输出层的输入
# 交叉熵损失函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(Y_pred), axis=1))
# 梯度下降
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
# 评估模型
correct_prediction = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype="float"))


def get_labels(labels):
    text_label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    temp = []
    for i in np.argmax(labels, 1):
        temp.append(text_label[i])
    return temp


with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)

    # 训练模型
    num_epochs = 1000
    for epoch in range(num_epochs):
        batch_xs, batch_ys = data_set.train.next_batch(batch_size=256)
        loss, trainer_ = sess.run([cross_entropy, train_step], feed_dict={X: batch_xs, y_: batch_ys})
        print('epoch', epoch+1, ', loss=', loss)

    # 测试模型
    # sample = data_set.test.images[533:552]
    # label = data_set.test.labels[533:552]
    # print('real labels:', get_labels(label))
    # pred_label = sess.run(Y_pred, feed_dict={X: sample, y_: label}).argmax(axis=1)
    # print('predictions:', pred_label)
    print('accuracy=', sess.run(accuracy, feed_dict={X: data_set.test.images, y_: data_set.test.labels}) * 100, "%")