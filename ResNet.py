from tensorflow import keras
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

train_images = mnist.train.images.reshape(mnist.train.images.shape[0], 28, 28, 1)
test_images = mnist.test.images.reshape(mnist.test.images.shape[0], 28, 28, 1)


# ResNet

def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same'):
    x = keras.layers.Convolution2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu')(x)
    x = keras.layers.BatchNormalization(axis=3)(x)
    return x


def residual_block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    # 进入A
    x = Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
    # 进入B
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
    # 如果经过A的输出和经过B的输出形状不同，则生成一个形状相同的输出
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
        x = keras.layers.add([x, shortcut])
        return x
    else:
        x = keras.layers.add([x, inpt])
        return x


def resnet_(width, height, channel, classes):
    inpt = keras.layers.Input(shape=(width, height, channel))
    x = keras.layers.ZeroPadding2D((3, 3))(inpt)

    # conv1
    x = Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
    x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # conv2_x
    x = residual_block(x, nb_filter=64, kernel_size=(3, 3))

    # conv3_x
    x = residual_block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)

    # conv4_x
    x = residual_block(x, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)

    # conv5_x
    x = residual_block(x, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)

    x = keras.layers.GlobalAvgPool2D()(x)
    x = keras.layers.Dense(classes, activation='softmax')(x)

    model = keras.models.Model(inputs=inpt, outputs=x)
    return model


IM_WIDTH = 28
IM_HEIGHT = 28
NB_CLASSES = 10
model = resnet_(IM_WIDTH, IM_HEIGHT, 1, NB_CLASSES)

lr = 0.05
num_epochs = 5
batch_size = 256
model.compile(optimizer=keras.optimizers.SGD(lr), loss='categorical_crossentropy', metrics=['acc'])

for epoch in range(num_epochs):
    history = model.fit(train_images, mnist.train.labels, batch_size=batch_size)
    print('epoch=', epoch + 1, '  loss=', history.history['loss'][0])
