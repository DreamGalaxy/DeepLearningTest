from tensorflow import keras
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

train_images = mnist.train.images.reshape(mnist.train.images.shape[0], 28, 28, 1)
test_images = mnist.test.images.reshape(mnist.test.images.shape[0], 28, 28, 1)

model = keras.Sequential()
# 第一层
model.add(keras.layers.Conv2D(filters=96, kernel_size=11, strides=4, activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=2, strides=2))
# 第二层
model.add(keras.layers.Conv2D(filters=256, kernel_size=5, padding='same', activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=2, strides=2))
# 第三层
model.add(keras.layers.Conv2D(filters=384, kernel_size=3, padding='same', activation='relu'))
# 第四层
model.add(keras.layers.Conv2D(filters=384, kernel_size=3, padding='same', activation='relu'))
# 第五层
model.add(keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=1, strides=2))
# Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。
model.add(keras.layers.Flatten())
# 第六层
model.add(keras.layers.Dense(4096, activation='relu'))
model.add(keras.layers.Dropout(0.5))
# 第七层
model.add(keras.layers.Dense(4096, activation='relu'))
model.add(keras.layers.Dropout(0.5))
# 第八层
model.add(keras.layers.Dense(10, activation='softmax'))


lr = 0.01
num_epochs = 5
batch_size = 128
model.compile(optimizer=keras.optimizers.SGD(lr), loss='categorical_crossentropy', metrics=['acc'])

for epoch in range(num_epochs):
    history = model.fit(train_images, mnist.train.labels, batch_size=batch_size)
    print('epoch=', epoch+1, '  loss=', history.history['loss'][0])
