import tensorflow as tf
import numpy as np
import random
from matplotlib import pyplot as plt
from IPython.display import set_matplotlib_formats


num_inputs = 2
num_examples = 1000
true_w = [2,-3.4]
true_b = 4.2
#features =tf.random_normal(shape=(num_examples, num_inputs),mean=0.0, stddev=1.0)
features = np.random.normal(scale=1, size=(num_examples, num_inputs))
labels = np.add(np.add(true_w[0] * features[:, 0], true_w[1] * features[:, 1]), true_b)
#labels += tf.random_normal(shape=[num_examples,1],mean=0.0,stddev=0.01)
labels += np.random.normal(scale=0.01, size=labels.shape)

def set_figsize(figsize=(3.5, 2.5)):
    set_matplotlib_formats('retina') # 打印高清图。
    plt.rcParams['figure.figsize'] = figsize # 设置图的尺寸。

set_figsize()
plt.scatter((features)[:, 1], labels, 1)
plt.show()


batch_size = 10
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 样本的读取顺序是随机的。
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = np.array(indices[i: min(i + batch_size, num_examples)])
        yield features[j,:], labels[j]

w = tf.Variable(tf.random_normal(shape=(num_inputs, 1),mean=0.0,stddev=0.01))
b = tf.Variable(tf.zeros(shape=(1,)))
X=tf.placeholder(tf.float32,[10,2])
y=tf.placeholder(tf.float32,[10,1])


lr = 0.03
num_epochs = 3
pred=tf.add(tf.matmul(X, w) , b)
loss=tf.losses.mean_squared_error(labels=y, predictions=pred)
optimizer=tf.train.GradientDescentOptimizer(lr).minimize(loss)


init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

# 训练模型一共需要 num_epochs 个迭代周期。
for epoch in range(1, num_epochs + 1):
    # 在一个迭代周期中，使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）。
    # X 和 y 分别是小批量样本的特征和标签。
    for tempX, tempY in data_iter(batch_size, features, labels):
        tempY=tempY.reshape(10,1)
        sess.run(optimizer, feed_dict={X: tempX, y: tempY})

    print('epoch %d, loss %f'% (epoch, sess.run(loss, {X: tempX, y: tempY})))


print(sess.run(w))
print(sess.run(b))