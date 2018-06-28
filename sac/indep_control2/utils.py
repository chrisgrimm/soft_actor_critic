import tensorflow as tf
import numpy as np
import os
import shutil

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


class TBDataWriter:

    def __init__(self):
        self.global_steps = dict()

    def setup(self, logdir):
        self.logdir = logdir
        self.summary_writer = tf.summary.FileWriter(self.logdir)

    def add_line(self, name, value):
        global_step = self.global_steps.get(name, 0)
        summary = tf.Summary()
        summary.value.add(tag=name, simple_value=value)
        self.summary_writer.add_summary(summary, global_step=global_step)
        self.global_steps[name] = global_step + 1
        self.summary_writer.flush()

    def purge(self):
        for name in os.listdir(self.logdir):
            path = os.path.join(self.logdir, name)
            if os.path.isfile(path):
                os.remove(path)



class ChoiceDataWriter:

    def __init__(self):
        pass

    def setup(self, mode, arg):
        if mode == 'tensorboard':
            self.logger = TBDataWriter()
            self.logger.setup(arg)
        elif mode == 'text':
            self.logger = DataWriter()
            self.logger.setup(arg)
        else:
            raise Exception(f'mode: {mode} unrecognized. mode must be in [\'tensorboard\', \'text\'].')

    def add_line(self, name, value):
        return self.logger.add_line(name, value)

    def purge(self):
        return self.logger.purge()

LOG = ChoiceDataWriter()

# credit: https://stackoverflow.com/questions/37086268/rename-variable-scope-of-saved-model-in-tensorflow
def ckpt_surgery(checkpoint_dir, modification_function, dry_run=False):
    if checkpoint_dir.endswith('.ckpt'):
        checkpoint_dir = os.path.split(checkpoint_dir)[0]
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    print(f'checkpoint {checkpoint}')
    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
            # Load the variable
            var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)
            # Set the new name
            new_name = modification_function(var_name)

            if dry_run:
                print('%s would be renamed to %s.' % (var_name, new_name))
            else:
                print('Renaming %s to %s.' % (var_name, new_name))
                # Rename the variable
                var = tf.Variable(var, name=new_name)

        if not dry_run:
            # Save the variables
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            saver.save(sess, checkpoint.model_checkpoint_path)

