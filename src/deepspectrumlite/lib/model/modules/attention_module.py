import tensorflow as tf
from tensorflow import keras

# code taken from https://github.com/kobiso/CBAM-tensorflow
# Author: Byung Soo Ko


def se_block(residual, name, ratio=8):
    """Contains the implementation of Squeeze-and-Excitation(SE) block.
      As described in https://arxiv.org/abs/1709.01507.
      """

    kernel_initializer = tf.keras.initializers.VarianceScaling()
    bias_initializer = tf.constant_initializer(value=0.0)

    with tf.name_scope(name):
        channel = residual.get_shape()[-1]
        # Global average pooling
        squeeze = tf.reduce_mean(residual, axis=[1, 2], keepdims=True)
        assert squeeze.get_shape()[1:] == (1, 1, channel)
        excitation = keras.layers.Dense(
                                     units=channel // ratio,
                                     activation=tf.nn.relu,
                                     kernel_initializer=kernel_initializer,
                                     bias_initializer=bias_initializer,
                                     name='bottleneck_fc')(squeeze)
        assert excitation.get_shape()[1:] == (1, 1, channel // ratio)
        excitation = keras.layers.Dense(
                                     units=channel,
                                     activation=tf.nn.sigmoid,
                                     kernel_initializer=kernel_initializer,
                                     bias_initializer=bias_initializer,
                                     name='recover_fc')(excitation)
        assert excitation.get_shape()[1:] == (1, 1, channel)
        # top = tf.multiply(bottom, se, name='scale')
        scale = residual * excitation
    return scale


def cbam_block(input_feature, name, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
  As described in https://arxiv.org/abs/1807.06521.
  """

    with tf.name_scope(name):
        attention_feature = channel_attention(input_feature, 'ch_at', ratio)
        attention_feature = spatial_attention(attention_feature, 'sp_at')
    return attention_feature


def channel_attention(input_feature, name, ratio=8):
    kernel_initializer = tf.keras.initializers.VarianceScaling()
    bias_initializer = tf.constant_initializer(value=0.0)

    with tf.name_scope(name):
        channel = input_feature.get_shape()[-1]
        avg_pool = tf.reduce_mean(input_feature, axis=[1, 2], keepdims=True)

        assert avg_pool.get_shape()[1:] == (1, 1, channel)
        avg_pool = keras.layers.Dense(
                                   units=channel // ratio,
                                   activation=tf.nn.relu,
                                   kernel_initializer=kernel_initializer,
                                   bias_initializer=bias_initializer)(avg_pool)
        assert avg_pool.get_shape()[1:] == (1, 1, channel // ratio)
        avg_pool = keras.layers.Dense(
                                   units=channel,
                                   kernel_initializer=kernel_initializer,
                                   bias_initializer=bias_initializer)(avg_pool)
        assert avg_pool.get_shape()[1:] == (1, 1, channel)

        max_pool = tf.reduce_max(input_feature, axis=[1, 2], keepdims=True)
        assert max_pool.get_shape()[1:] == (1, 1, channel)
        max_pool = keras.layers.Dense(
                                   units=channel // ratio,
                                   activation=tf.nn.relu)(max_pool)
        assert max_pool.get_shape()[1:] == (1, 1, channel // ratio)
        max_pool = keras.layers.Dense(
                                   units=channel)(max_pool)
        assert max_pool.get_shape()[1:] == (1, 1, channel)

        scale = tf.sigmoid(avg_pool + max_pool, 'sigmoid')

    return input_feature * scale


def spatial_attention(input_feature, name):
    kernel_size = 7
    kernel_initializer = tf.keras.initializers.VarianceScaling()
    with tf.name_scope(name):
        avg_pool = tf.reduce_mean(input_feature, axis=[3], keepdims=True)
        assert avg_pool.get_shape()[-1] == 1
        max_pool = tf.reduce_max(input_feature, axis=[3], keepdims=True)
        assert max_pool.get_shape()[-1] == 1
        concat = tf.concat([avg_pool, max_pool], 3)
        assert concat.get_shape()[-1] == 2

        concat = keras.layers.Conv2D(
                                  filters=1,
                                  kernel_size=[kernel_size, kernel_size],
                                  strides=[1, 1],
                                  padding="same",
                                  activation=None,
                                  kernel_initializer=kernel_initializer,
                                  use_bias=False)(concat)
        assert concat.get_shape()[-1] == 1
        concat = tf.sigmoid(concat, 'sigmoid')

    return input_feature * concat
