import tensorflow as tf
import numpy as np
import os

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


def build_directory_structure(base_dir, dir_structure):
    current_path = base_dir
    for target_key in dir_structure.keys():
        target_path = os.path.join(current_path, target_key)
        # make the dir if it doesnt exist.
        if not os.path.isdir(target_path):
            os.mkdir(target_path)
        # build downwards
        build_directory_structure(target_path, dir_structure[target_key])


class DataWriter:

    # file_mapping maps names to file paths
    def __init__(self):
        self.file_mapping = None

    def setup(self, file_mapping):
        self.file_mapping = file_mapping

    def add_line(self, name, value):
        if self.file_mapping is None:
            raise Exception('DataWriter unitialized, please call `setup` before using.')
        if name not in self.file_mapping:
            raise Exception(f'Name "{name}" not in file mapping.')
        with open(self.file_mapping[name], 'a') as f:
            f.write(str(value)+'\n')

    def purge(self):
        if self.file_mapping is None:
            raise Exception('DataWriter unitialized, please call `setup` before using.')
        for name in self.file_mapping:
            with open(self.file_mapping[name], 'w') as f:
                pass


LOG = DataWriter()