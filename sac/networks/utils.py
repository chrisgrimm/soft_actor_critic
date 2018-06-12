import numpy as np
import tensorflow as tf

def power2_encoding(s):
    shape = [x.value for x in s.get_shape()[1:]]
    is_pow2 = lambda x: x & (x - 1) == 0
    assert len(shape) == 3
    assert shape[0] == shape[1]
    assert is_pow2(shape[0])
    num_steps = int(np.round(np.log2(shape[0])))
    # conv down until we hit 8x8
    ci = s
    for i in range(num_steps - 3):
        ci = tf.layers.conv2d(ci, 32, 5, activation=tf.nn.relu, strides=2, padding='SAME', name=f'c{i}')
    # whatever we're left with should be 8x8 or lower.
    new_size = np.minimum(8, ci.get_shape()[1].value)
    print(ci)
    print(new_size)
    flat = tf.reshape(ci, [-1, new_size * new_size * 32])
    enc = tf.layers.dense(flat, 128, activation=tf.nn.relu, name='fc1')
    return enc
