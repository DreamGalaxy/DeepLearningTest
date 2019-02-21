import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_Fashion/", one_hot=True)
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 丢弃函数
def dropout(h, drop_prob):
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob
    # 这种情况下把全部元素都丢弃。
    if keep_prob == 0:
        return tf.zeros_like(h)
    mask = tf.random_uniform(tf.shape(h), minval=0, maxval=1) < keep_prob
    # tf.where的特殊用法，tf.where(input, a,b)，作用是a中对应input中true的位置的元素值不变，其余元素进行替换，替换成b中对应位置的元素值
    return tf.where(mask, h/keep_prob, tf.zeros(shape=tf.shape(h)))

num_inputs = 784
num_outputs = 10
num_hiddens1 = 256
num_hiddens2 = 256

W1 = tf.Variable(tf.random_normal(stddev=0.01, shape=(num_inputs, num_hiddens1)))
b1 = tf.Variable(tf.zeros(num_hiddens1))
W2 = tf.Variable(tf.random_normal(stddev=0.01, shape=(num_hiddens1, num_hiddens2)))
b2 = tf.Variable(tf.zeros(num_hiddens2))
W3 = tf.Variable(tf.random_normal(stddev=0.01, shape=(num_hiddens2, num_outputs)))
b3 = tf.Variable(tf.zeros(num_outputs))

drop_prob1 = 0.2
drop_prob2 = 0.5

X = tf.placeholder(tf.float32, [None, num_inputs])
y_true = tf.placeholder(tf.float32, [None, num_outputs])

H1_train = tf.nn.relu(tf.matmul(X, W1)+b1)
H1_train = dropout(H1_train, drop_prob1)
H2_train = tf.nn.relu(tf.matmul(H1_train, W2)+b2)
H2_train = dropout(H2_train, drop_prob2)
y_train_pred = tf.nn.softmax(tf.matmul(H2_train, W3)+b3)

H1_test = tf.nn.relu(tf.matmul(X, W1)+b1)
H2_test = tf.nn.relu(tf.matmul(H1_test, W2)+b2)
y_test_pred = tf.nn.softmax(tf.matmul(H2_test, W3)+b3)

batch_size = 256
num_epochs = 10
lr = 0.5

loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=tf.matmul(H2_train, W3)+b3))/batch_size
train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_test_pred, 1), tf.argmax(y_true, 1)), "float"))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    train_ls = []
    test_ls = []

    for epoch in range(1, num_epochs + 1):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        train_l, train_ = sess.run([loss, train_step], feed_dict={X: batch_xs, y_true: batch_ys})
        print('epoch', epoch, 'loss=', train_l)
        train_ls.append(train_l)
        test_l = sess.run(loss, feed_dict={X:  mnist.test.images, y_true: mnist.test.labels})
        test_ls.append(test_l)

    print('准确率:', sess.run(accuracy, feed_dict={X: mnist.test.images, y_true: mnist.test.labels}) * 100, '%')
