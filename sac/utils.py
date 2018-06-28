import numpy as np
import tensorflow as tf
import GPUtil
import warnings

def onehot(idx, num_entries):
    x = np.zeros(num_entries)
    x[idx] = 1
    return x

def horz_stack_images(*images, spacing=5, background_color=(0,0,0)):
    # assert that all shapes have the same siz
    if len(set([tuple(image.shape) for image in images])) != 1:
        raise Exception('All images must have same shape')
    if images[0].shape[2] != len(background_color):
        raise Exception('Depth of background color must be the same as depth of image.')
    height = images[0].shape[0]
    width = images[0].shape[1]
    depth = images[0].shape[2]
    canvas = np.ones([height, width*len(images) + spacing*(len(images) - 1), depth])
    bg_color = np.reshape(background_color, [1, 1, depth])
    canvas *= bg_color
    width_pos = 0
    for image in images:
        canvas[:, width_pos:width_pos+width, :] = image
        width_pos += (width + spacing)
    return canvas


def component(function):

    def wrapper(*args, **kwargs):
        reuse = kwargs.get('reuse', None)
        name = kwargs['name']
        if 'reuse' in kwargs:
            del kwargs['reuse']
        del kwargs['name']
        with tf.variable_scope(name, reuse=reuse):
            out = function(*args, **kwargs)
            variables = tf.get_variable_scope().get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            return out, variables
    return wrapper

def get_best_gpu():
    return GPUtil.getAvailable(order='load', limit=10, maxLoad=1.0, maxMemory=1.0)[0]


class HyperParams(object):

    def __init__(self):
        self.params = dict()
        self.used_params = set()


    def param(self, item, reuse=False):
        if item in self.params:
            if item in self.used_params and not reuse:
                raise Exception(f'Parameter {item} previously used, but reuse is set to False.')
            self.used_params.add(item)
            return self.params[item]
        else:
            raise Exception(f'Unspecified hyperparameter key: {item}')


