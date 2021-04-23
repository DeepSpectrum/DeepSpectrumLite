import tensorflow as tf


# Based on the typescript implemented
# @see https://github.com/alesgenova/colormap


def linear_scale(domain, out_range, value): # pragma: no cover
    d0, d1 = domain
    r0, r1 = out_range

    return r0 + (r1 - r0) * ((value - d0) / (d1 - d0))


def linear_mixer(value, lower_node_value, upper_node_value): # pragma: no cover
    frac = (value - lower_node_value) / (upper_node_value - lower_node_value)
    return 1. - frac, frac


def color_combination(a, X, b, Y): # pragma: no cover
    return [
        a * X[0] + b * Y[0],
        a * X[1] + b * Y[1],
        a * X[2] + b * Y[2]
    ]


def create_map_from_array(v, color_map): # pragma: no cover
    channel_position = 3
    # rgb:
    v = tf.expand_dims(v, channel_position)  # add channel axis
    v = tf.repeat(v, 3, axis=channel_position)  # duplicate values to all channels

    domain = (tf.math.reduce_min(v, axis=(1, 2, 3), keepdims=True),
              tf.math.reduce_max(v, axis=(1, 2, 3), keepdims=True))

    scaled_value = linear_scale(domain=domain, out_range=(0, 1), value=v)
    vri_len = color_map.shape[0]
    index_float = (vri_len - 1) * scaled_value

    t1 = tf.math.less_equal(index_float, 0)
    t2 = tf.math.greater_equal(index_float, vri_len - 1)

    result = tf.where(t1, color_map[0], tf.where(t2, color_map[vri_len - 1], [-1, -2, -3]))

    index = tf.math.floor(index_float)
    index2 = tf.where(index >= vri_len - 1, tf.cast(vri_len, dtype=tf.float32) - 1, index + 1.)

    coeff0, coeff1 = linear_mixer(value=index_float, lower_node_value=index, upper_node_value=index2)
    index = tf.cast(index, dtype=tf.int32)
    index2 = tf.cast(index2, dtype=tf.int32)

    # red mask
    v1_r = tf.gather(color_map[:, 0], indices=index)
    v2_r = tf.gather(color_map[:, 0], indices=index2)
    com_r = coeff0 * v1_r + coeff1 * v2_r
    # green mask
    v1_g = tf.gather(color_map[:, 1], indices=index)
    v2_g = tf.gather(color_map[:, 1], indices=index2)
    com_g = coeff0 * v1_g + coeff1 * v2_g
    # blue mask
    v1_b = tf.gather(color_map[:, 2], indices=index)
    v2_b = tf.gather(color_map[:, 2], indices=index2)
    com_b = coeff0 * v1_b + coeff1 * v2_b
    # apply color masks
    result = tf.where(tf.math.equal(result, -1), com_r, result)
    result = tf.where(tf.math.equal(result, -2), com_g, result)
    result = tf.where(tf.math.equal(result, -3), com_b, result)

    return result
