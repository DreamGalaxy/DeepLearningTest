# -*- coding: UTF-8 -*-

# 使用的是keras版本的GoogLeNet，因为tensorflow.keras没有办法将模型
# 的参数通过tf.keras.models.Model整合从而构造出模型，也无法通过
# model.add进行简单网络的叠加

from tensorflow import keras
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

train_images = mnist.train.images.reshape(mnist.train.images.shape[0], 28, 28, 1)
test_images = mnist.test.images.reshape(mnist.test.images.shape[0], 28, 28, 1)


# GooLeNet
# 定义模型

def inception_module(x, params, concat_axis, padding='same', activation='relu'):
    (branch1, branch2, branch3, branch4) = params
    # 1x1
    pathway1 = keras.layers.Convolution2D(filters=branch1[0], kernel_size=(1, 1), activation=activation)(x)
    # 1x1->3x3
    pathway2 = keras.layers.Convolution2D(filters=branch2[0], kernel_size=(1, 1), activation=activation)(x)
    pathway2 = keras.layers.Convolution2D(filters=branch2[1], kernel_size=(3, 3),
                             padding=padding, activation=activation)(pathway2)
    # 1x1->5x5
    pathway3 = keras.layers.Convolution2D(filters=branch3[0], kernel_size=(1, 1), activation=activation)(x)
    pathway3 = keras.layers.Convolution2D(filters=branch3[1], kernel_size=(5, 5),
                             padding=padding, activation=activation)(pathway3)
    # 3x3->1x1
    pathway4 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=1, padding=padding)(x)
    pathway4 = keras.layers.Convolution2D(filters=branch4[0], kernel_size=(1, 1), activation=activation)(pathway4)
    return keras.layers.concatenate([pathway1, pathway2, pathway3, pathway4], axis=concat_axis)


def create_model():
    INP_SHAPE = (28, 28, 1)
    img_input = keras.Input(shape=INP_SHAPE)
    CONCAT_AXIS = 3
    NB_CLASS = 10

    # module 1
    x = keras.layers.Convolution2D(filters=64, kernel_size=(7, 7), strides=2, padding='same')(img_input)
    x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    # 为防止梯度弥散，在网络的第一模块最后加入批量规范化层可以避免随机性可能造成的梯度弥散，使精度停留在0.1左右
    x = keras.layers.BatchNormalization()(x)

    # module 2
    x = keras.layers.Convolution2D(filters=64, kernel_size=(1, 1))(x)
    x = keras.layers.Convolution2D(filters=192, kernel_size=(3, 3), padding='same')(x)
    x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    # module 3
    x = inception_module(x, params=[(64,), (96, 128), (16, 32), (32,)], concat_axis=CONCAT_AXIS)
    x = inception_module(x, params=[(128,), (128, 192), (32, 96), (64,)], concat_axis=CONCAT_AXIS)
    x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2)(x)

    # module 4
    x = inception_module(x, params=[(192,), (96, 208), (16, 48), (64,)], concat_axis=CONCAT_AXIS)
    x = inception_module(x, params=[(160,), (112, 224), (24, 64), (64,)], concat_axis=CONCAT_AXIS)
    x = inception_module(x, params=[(128,), (128, 256), (24, 64), (64,)], concat_axis=CONCAT_AXIS)
    x = inception_module(x, params=[(112,), (144, 288), (32, 64), (64,)], concat_axis=CONCAT_AXIS)
    x = inception_module(x, params=[(256,), (160, 320), (32, 128), (128,)], concat_axis=CONCAT_AXIS)
    x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    # module 5
    x = inception_module(x, params=[(256,), (160, 320), (32, 128), (128,)], concat_axis=CONCAT_AXIS)
    x = inception_module(x, params=[(384,), (192, 384), (48, 128), (128,)], concat_axis=CONCAT_AXIS)
    x = keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same')(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=NB_CLASS, activation='softmax')(x)

    return x, img_input, CONCAT_AXIS, INP_SHAPE


# Create the Model
x, img_input, CONCAT_AXIS, INP_SHAPE = create_model()
# Create a Keras Model
model = keras.Model(inputs=img_input, outputs=[x])

lr = 0.1
num_epochs = 5
batch_size = 128
model.compile(optimizer=keras.optimizers.SGD(lr), loss='categorical_crossentropy', metrics=['acc'])

for epoch in range(num_epochs):
    history = model.fit(train_images, mnist.train.labels, batch_size=batch_size)
    print('epoch=', epoch + 1, '  loss=', history.history['loss'][0])

