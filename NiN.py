from tensorflow import keras
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

train_images = mnist.train.images.reshape(mnist.train.images.shape[0], 28, 28, 1)
test_images = mnist.test.images.reshape(mnist.test.images.shape[0], 28, 28, 1)


def nin_block(num_channels, kernel_size, strides, padding):
    model.add(keras.layers.Conv2D(num_channels, kernel_size, strides, padding, activation='relu'))
    model.add(keras.layers.Conv2D(num_channels, kernel_size=1, activation='relu'))
    model.add(keras.layers.Conv2D(num_channels, kernel_size=1, activation='relu'))

model = keras.Sequential()
nin_block(96, kernel_size=11, strides=4, padding='valid')
model.add(keras.layers.MaxPool2D(pool_size=2, strides=2))
nin_block(256, kernel_size=5, strides=1, padding='same')
model.add(keras.layers.MaxPool2D(pool_size=2, strides=2))
nin_block(384, kernel_size=3, strides=1, padding='same')
model.add(keras.layers.MaxPool2D(pool_size=1, strides=2))
model.add(keras.layers.Dropout(0.5))
# 标签类数是 10。
nin_block(10, kernel_size=3, strides=1, padding='same')
# 全局平均池化层将窗口形状自动设置成输出的高和宽
model.add(keras.layers.GlobalAvgPool2D())
model.add(keras.layers.Activation('softmax'))
# 将四维的输出转成二维的输出，其形状为（批量大小，10）
model.add(keras.layers.Flatten())

lr = 0.1
num_epochs = 5
batch_size = 128
model.compile(optimizer=keras.optimizers.SGD(lr), loss='categorical_crossentropy', metrics=['acc'])

for epoch in range(num_epochs):
    history = model.fit(train_images, mnist.train.labels, batch_size=batch_size)
    print('epoch=', epoch + 1, '  loss=', history.history['loss'][0])
