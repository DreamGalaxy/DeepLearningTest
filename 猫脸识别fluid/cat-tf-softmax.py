import numpy as np

import lr_utils
import tensorflow as tf

learning_rate = 0.008
training_epochs = 2000
batch_size = 40
display_step = 1

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = lr_utils.load_dataset()
print(train_set_x_orig.shape)
m_train,m_test,num_px = train_set_x_orig.shape[0],test_set_x_orig.shape[0],train_set_x_orig.shape[1]
print(m_train,m_test,num_px)
DATA_DIM = num_px * num_px * 3
train_set_x_flatten = (train_set_x_orig/255).reshape(m_train,DATA_DIM)
# train_set_x_flatten = train_set_x_flatten/255
test_set_x_flatten = (test_set_x_orig/255).reshape(m_test,DATA_DIM)
# test_set_x_flatten = test_set_x_flatten/255
train_set_y = train_set_y.reshape(m_train,1)
train_set = 1 - train_set_y
train = np.hstack((train_set_y,train_set))
print(train.shape)
test_set_y = test_set_y.reshape(m_test,1)
test_set = 1 - test_set_y
test = np.hstack((test_set_y,test_set))
print(test_set_y.shape,train_set_x_flatten.shape)
# tf Graph Input
x = tf.placeholder(tf.float32, [None, DATA_DIM])
y = tf.placeholder(tf.float32, [None, 2])

# Set model weights
W = tf.Variable(tf.zeros([DATA_DIM, 2]))
b = tf.Variable(tf.zeros([2]))

# softmax
pred = tf.nn.softmax(tf.matmul(x, W) + b)

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(m_train/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = train_set_x_flatten[i*batch_size:(i+1)*batch_size],train[i*batch_size:(i+1)*batch_size]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x:test_set_x_flatten, y: test}))

