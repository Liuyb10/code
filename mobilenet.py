from keras import backend
from keras.layers import *
from keras.models import *
from keras import backend as K
import tensorflow as tf
from utils.metrics import f_score
K.set_learning_phase(True)
if tf.__version__>= '2.0.0':
    tf.compat.v1.disable_eager_execution()
def _activation(x, name='relu'):
    if name == 'relu':
        return Activation('relu')(x)
    elif name == 'hardswish':
        return hard_swish(x)


def hard_sigmoid(x):
    return backend.relu(x + 3.0, max_value=6.0) / 6.0


def hard_swish(x):
    return Multiply()([Activation(hard_sigmoid)(x), x])


def cbam_block(cbam_feature, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """

    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature


def channel_attention(input_feature, ratio=8):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]

    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel // ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel // ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])


def spatial_attention(input_feature):
    kernel_size = 7

    if K.image_data_format() == "channels_first":
        channel = input_feature.shape[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature.shape[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool.shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool.shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat.shape[-1] == 2
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    assert cbam_feature.shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])


def relu6(x):
    return K.relu(x, max_value=6)


def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1), dilation_rate=(1, 1)):
    filters = int(filters * alpha)
    x = Conv2D(filters, kernel, padding='same',
               use_bias=False,
               dilation_rate=dilation_rate,
               strides=strides,
               name='conv1')(inputs)
    x = BatchNormalization(name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)

def resblock(inputs, out_channel, strides=(1,1)):

    x = _conv_block(inputs, out_channel, 1, strides=strides, dilation_rate=(1, 1))
    x = _conv_block(x, out_channel, 1, strides=strides, dilation_rate=(1, 1))
    x = x + inputs
    return x

def _depthwise_conv_block(inputs, pointwise_conv_filters, up_dim, depth_multiplier=1, se=False, strides=(1, 1),
                          dilation_rate=(1, 1), activation='relu', block_id=1):
    skip_flag = strides == (1, 1) and inputs[3] == pointwise_conv_filters

    x = Conv2D(int(up_dim), (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw1_%d' % block_id)(inputs)

    x = DepthwiseConv2D((3, 3), padding='same',
                        depth_multiplier=depth_multiplier,
                        dilation_rate=dilation_rate,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(x)
    x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
    x = _activation(x, activation)
    if se:
        x = cbam_block(x)
    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw2_%d' % block_id)(x)
    x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    x = _activation(x, activation)

    if skip_flag:
        x = Add()([x, inputs])
    return x


def get_mobilenet_encoder(input_height=320, input_width=320):
    alpha = 1.0

    img_input = Input(shape=(input_height, input_width, 3))

    # 320,320,3 -> 160,160,16
    x = _conv_block(img_input, 16, alpha, strides=(2, 2), dilation_rate=(1, 1))
    # 160,160,16 -> 80 80 16
    x = _depthwise_conv_block(x, 16, up_dim=16, depth_multiplier=1, se=False, strides=(2, 2), dilation_rate=(1, 1), activation='relu',
                              block_id=1)
    f1 = x

    # 80 80 16 -> 40 40 24
    x = _depthwise_conv_block(x, 24, up_dim=72, depth_multiplier=1, se=False, strides=(2, 2), dilation_rate=(3, 3),
                              activation='relu',
                              block_id=2)
    # 40 40 24 -> 40 40 24
    x = _depthwise_conv_block(x, 24, up_dim=88, depth_multiplier=1, se=False, dilation_rate=(5, 5), activation='relu',
                              block_id=3)

    # 40 40 24 -> 20 20 40
    x = _depthwise_conv_block(x, 40, up_dim=96, depth_multiplier=1, se=False, strides=(2, 2), dilation_rate=(1, 1),
                              activation='relu',
                              block_id=4)
    # 20 20 40 -> 20 20 40
    x = _depthwise_conv_block(x, 40, up_dim=240, depth_multiplier=1, se=False, dilation_rate=(3, 3), activation='relu',
                              block_id=5)

    # 20 20 40 -> 20 20 40
    x = _depthwise_conv_block(x, 40, up_dim=240, depth_multiplier=1, se=False, dilation_rate=(5, 5),
                              activation='hardswish', block_id=6)
    # 20 20 40 -> 20 20 48
    x = _depthwise_conv_block(x, 48, up_dim=120, depth_multiplier=1, se=True, dilation_rate=(1, 1), activation='hardswish',
                              block_id=7)
    # 20 20 48 -> 20 20 48
    x = _depthwise_conv_block(x, 48, up_dim=144, depth_multiplier=1, se=True, dilation_rate=(3, 3), activation='hardswish',
                              block_id=8)
    # 20 20 40 -> 10 10 96
    x = _depthwise_conv_block(x, 96, up_dim=288, depth_multiplier=1, se=True, strides=(2, 2), dilation_rate=(5, 5), activation='hardswish',
                              block_id=9)
    # 10 10 96 -> 10 10 96
    x = _depthwise_conv_block(x, 96, up_dim=576, depth_multiplier=1, se=True, dilation_rate=(1, 1), activation='hardswish',
                              block_id=10)
    # 10 10 96 -> 10 10 96
    x = _depthwise_conv_block(x, 96, up_dim=576, depth_multiplier=1, se=True, dilation_rate=(3, 3), activation='hardswish',
                              block_id=11)


    return img_input, f1, x
