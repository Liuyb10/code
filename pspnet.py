import tensorflow as tf
from keras.layers import *
from keras.models import *
import numpy as np
from keras import backend as K
from net_conv5.mobilenet import get_mobilenet_encoder
from net_conv5.mobilenet import _depthwise_conv_block, _activation


def resize_image(inp, s):
    return Lambda(lambda x: tf.image.resize(x, (K.int_shape(x)[1] * s[0], K.int_shape(x)[2] * s[1])))(inp)


def conv2d_bn(x, filters, num_row, num_col, padding='same', stride=1, dilation_rate=1, relu=True):
    x = Conv2D(
        filters, (num_row, num_col),
        strides=(stride, stride),
        padding=padding,
        dilation_rate=(dilation_rate, dilation_rate),
        use_bias=False)(x)
    x = BatchNormalization()(x)
    if relu:
        x = _activation(x, 'relu')
    return x

def BasicRFB(x, input_filters, output_filters, stride=1, map_reduce=8):
    # -------------------------------------------------------#
    #   BasicRFB模块是一个残差结构
    #   主干部分使用不同膨胀率的卷积进行特征提取
    #   残差边只包含一个调整宽高和通道的1x1卷积
    # -------------------------------------------------------#
    # input_filters = 96    .   input_filters_div =12
    input_filters_div = input_filters // map_reduce

    branch_0 = conv2d_bn(x, input_filters_div * 2, 1, 1, stride=stride)
    branch_0 = conv2d_bn(branch_0, input_filters_div * 2, 3, 3, relu=False)

    branch_1 = conv2d_bn(x, input_filters_div, 1, 1)
    branch_1 = conv2d_bn(branch_1, input_filters_div * 2, 3, 3, stride=stride)
    branch_1 = conv2d_bn(branch_1, input_filters_div * 2, 3, 3, dilation_rate=3, relu=False)

    branch_2 = conv2d_bn(x, input_filters_div, 1, 1)
    branch_2 = conv2d_bn(branch_2, (input_filters_div // 2) * 3, 3, 3)
    branch_2 = conv2d_bn(branch_2, input_filters_div * 2, 3, 3, stride=stride)
    branch_2 = conv2d_bn(branch_2, input_filters_div * 2, 3, 3, dilation_rate=5, relu=False)

    branch_3 = conv2d_bn(x, input_filters_div, 1, 1)
    branch_3 = conv2d_bn(branch_3, (input_filters_div // 2) * 3, 1, 7)
    branch_3 = conv2d_bn(branch_3, input_filters_div * 2, 7, 1, stride=stride)
    branch_3 = conv2d_bn(branch_3, input_filters_div * 2, 3, 3, dilation_rate=7, relu=False)

    # -------------------------------------------------------#
    #   将不同膨胀率的卷积结果进行堆叠
    #   利用1x1卷积调整通道数
    # -------------------------------------------------------#
    out = concatenate([branch_0, branch_1, branch_2, branch_3], axis=-1)
    out = conv2d_bn(out, output_filters, 1, 1, relu=False)

    # -------------------------------------------------------#
    #   残差边也需要卷积，才可以相加
    # -------------------------------------------------------#
    short = conv2d_bn(x, output_filters, 1, 1, stride=stride, relu=False)
    out = Lambda(lambda x: x[0] + x[1])([out, short])
    out = _activation(out, 'relu')
    return out

def pool_block(feats, pool_factor):
    h = K.int_shape(feats)[1]
    w = K.int_shape(feats)[2]

    # -----------------------------------------------------#
    # 	strides = [18, 18],[9, 9],[6, 6],[3, 3]
    #   1    2    5      10
    #   10   5    2      1
    # 	进行不同程度的平均
    # -----------------------------------------------------#
    pool_size = strides = [int(np.round(float(h) / pool_factor)), int(np.round(float(w) / pool_factor))]
    x = AveragePooling2D(pool_size, strides=strides, padding='same')(feats)

    # -----------------------------------------------------#
    #   利用1x1卷积进行通道数的调整
    # -----------------------------------------------------#
    x = Conv2D(512, (1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = _activation(x, 'relu')

    # -----------------------------------------------------#
    #   利用resize扩大特征层面积
    # -----------------------------------------------------#
    x = resize_image(x, strides)
    return x


def _pspnet(n_classes, encoder, input_height=320, input_width=320):
    # f5     10,10,96
    # f1     80,80,16

    img_input, feate, x = encoder(input_height=input_height, input_width=input_width)
    x = x

    # 10,10,96->10,10,96
    x = BasicRFB(x, 96, 96)
    print(K.int_shape(x))

    # 10,10,96 -> 10, 10, 96
    x = _depthwise_conv_block(x, 96, up_dim=576, depth_multiplier=1, se=True, dilation_rate=(5, 5), activation='hardswish',
                              block_id=12)

    print(K.int_shape(x))
    print(K.int_shape(feate))

    # 10,10,96 -> 80 80 96
    print(feate.shape[1:3])
    x = Lambda(lambda xx: tf.image.resize(x, [80, 80]))(x)
    print(K.int_shape(x))

    # 80,80,16 -> 80 80 32
    feate = Lambda(lambda f2f2: tf.image.resize(feate, [80, 80]))(feate)
    feate = Conv2D(32, (1, 1), padding='same', use_bias=False, name='feature_projection0')(feate)
    feate = BatchNormalization(name='feature_projection0_BN', epsilon=1e-5)(feate)
    feate = Activation('relu')(feate)
    print(K.int_shape(feate))

    # 80 80 96 + 80 80 32 = 80 80 128
    x = concatenate([x, feate], axis=3)

    x = _depthwise_conv_block(x, 96, up_dim=576, depth_multiplier=1, se=True, dilation_rate=(1, 1), activation='hardswish',
                              block_id=13)
    x = _depthwise_conv_block(x, 96, up_dim=576, depth_multiplier=1, se=True, dilation_rate=(3, 3), activation='hardswish',
                              block_id=14)

    # 80 80 128 -> 80, 80, nclasses
    x = Conv2D(n_classes, (3, 3), padding='same', dilation_rate=(1, 1))(x)
    x = BatchNormalization(name='feature_projection1_BN', epsilon=1e-5)(x)
    x = _activation(x, 'relu')

    size_before3 = tf.keras.backend.int_shape(img_input)
    x = Lambda(lambda xx: tf.image.resize(xx, size_before3[1:3]))(x)
    print('上采样输出', K.int_shape(x))

    # o = resize_image(o, (8,8))

    x = Reshape((-1, n_classes))(x)
    x = Softmax()(x)
    print('输出', K.int_shape(x))
    model = Model(img_input, x)
    return model


def mobilenet_pspnet(n_classes, input_height=320, input_width=320):
    model = _pspnet(n_classes, get_mobilenet_encoder, input_height=input_height, input_width=input_width)
    model.model_name = "mobilenet_pspnet"
    return model
