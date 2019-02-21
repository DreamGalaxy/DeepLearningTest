from tensorflow import keras
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_Fashion/", one_hot=True)
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

train_images = mnist.train.images.reshape(mnist.train.images.shape[0], 28, 28, 1)
test_images = mnist.test.images.reshape(mnist.test.images.shape[0], 28, 28, 1)


# 稠密块
def conv_block(x, num_channels, padding='same'):
    x = keras.layers.BatchNormalization(axis=3)(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Convolution2D(num_channels, kernel_size=3, padding=padding)(x)
    return x


def dense_block(inpt, num_convs, num_channels):
    x = conv_block(inpt, num_channels, padding='same')
    for num in range(num_convs-1):
        x = conv_block(x, num_channels, padding='same')
    x = keras.layers.concatenate([x, inpt], axis=3)
    return x


# 过渡块
def transition_block(inpt, num_channels):
    x = keras.layers.BatchNormalization(axis=3)(inpt)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Convolution2D(num_channels, kernel_size=1)(x)
    x = keras.layers.AvgPool2D(pool_size=1, strides=1)(x)
    return x


# DenseNet模型
def densenet(width, height, channel, classes):
    inpt = keras.layers.Input(shape=(width, height, channel))
    x = keras.layers.ZeroPadding2D((3, 3))(inpt)

    # conv1
    x = keras.layers.Convolution2D(64, kernel_size=(7, 7), strides=(2, 2), padding='valid')(x)
    x = keras.layers.BatchNormalization(axis=3)(x)
    x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # conv2_x
    x = dense_block(x, num_convs=4, num_channels=32)
    x = transition_block(x, num_channels=96)

    # conv3_x
    x = dense_block(x, num_convs=4, num_channels=32)
    x = transition_block(x, num_channels=160)

    # conv4_x
    x = dense_block(x, num_convs=4, num_channels=32)
    x = transition_block(x, num_channels=224)

    # conv5_x
    x = dense_block(x, num_convs=4, num_channels=32)

    x = keras.layers.BatchNormalization(axis=3)(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.GlobalAvgPool2D()(x)
    x = keras.layers.Dense(classes, activation='softmax')(x)

    model = keras.models.Model(inputs=inpt, outputs=x)
    return model

IM_WIDTH = 28
IM_HEIGHT = 28
NB_CLASSES = 10
model = densenet(IM_WIDTH, IM_HEIGHT, 1, NB_CLASSES)

lr = 0.05
num_epochs = 5
batch_size = 256
model.compile(optimizer=keras.optimizers.SGD(lr), loss='categorical_crossentropy', metrics=['acc'])

for epoch in range(num_epochs):
    history = model.fit(train_images, mnist.train.labels, batch_size=batch_size)
    print('epoch=', epoch + 1, '  loss=', history.history['loss'][0])
