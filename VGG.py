from tensorflow import keras
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

train_images = mnist.train.images.reshape(mnist.train.images.shape[0], 28, 28, 1)
test_images = mnist.test.images.reshape(mnist.test.images.shape[0], 28, 28, 1)


def vgg_block(num_convs, num_channels):
    for _ in range(num_convs):
        model.add(keras.layers.Conv2D(
            num_channels, kernel_size=3, activation='relu'))
    # model.add(keras.layers.MaxPool2D(pool_size=2, strides=2))


conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]


model = keras.Sequential()
# 卷积层部分。
for (num_convs, num_channels) in conv_arch:
    vgg_block(num_convs, num_channels)
# Flatten层过度
model.add(keras.layers.Flatten())
# 全连接层部分。
model.add(keras.layers.Dense(4096, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(4096, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10))

lr = 0.05
num_epochs = 5
batch_size = 128
model.compile(optimizer=keras.optimizers.SGD(lr), loss='categorical_crossentropy', metrics=['acc'])

for epoch in range(num_epochs):
    history = model.fit(train_images, mnist.train.labels, batch_size=batch_size)
    print('epoch=', epoch + 1, '  loss=', history.history['loss'][0])
