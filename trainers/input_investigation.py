import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
import tensorflow as tf
import os
from scipy.io import savemat


class InputDistInvestigator(object):
    def __init__(self, model, data, config):
        """
        This is a simple routine for the investigation of the learned input distribution
        """
        self.config = config
        self.model = model
        self.channel_name = config.channel_name
        self.load_enc_weights()
        self.data_iterator = data
        self.bptt = config.bptt
        self.data_batches = config.data_batches
        self.feedback = config.feedback

        if self.feedback:
            self.fb_input = tf.zeros(shape=[config.batch_size, 1, config.x_dim + config.y_dim])
        else:
            self.fb_input = tf.zeros(shape=[config.batch_size, 1, config.x_dim])

    def load_enc_weights(self):
        """
        set pretrained encoder weights
        """
        # filepath = os.path.join("./pretrained_models", self.config.channel_name + "_fb", "p_05_best_so_far", "weights")
        filepath = os.path.join("./pretrained_models", self.config.channel_name + "_ff", "new0", "weights")
        self.model['enc'].load_weights(filepath)

    def train(self):
        data = self.generate_long_sequence()
        self.save_sequence(data)

    def generate_long_sequence(self):
        data = {
            'x': [],
            'y': [],
            'p': [],
            's': []
        }
        for i in range(self.data_batches):
            [x, y, p, s] = self.model['enc'](self.fb_input)

            x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
            y_ = tf.split(y, axis=1, num_or_size_splits=self.config.bptt)[-1]

            if self.feedback:
                self.fb_input = tf.concat([x_, y_], axis=-1)
            else:
                self.fb_input = x_

            data['x'].append(x.numpy())
            data['y'].append(y.numpy())
            data['p'].append(p.numpy())
            data['s'].append(s.numpy())
            print(f"collected {i*self.config.bptt} samples")

        data['x'] = np.concatenate(data['x'], axis=1)[0:14,:,:]
        data['y'] = np.concatenate(data['y'], axis=1)[0:14,:,:]
        data['p'] = np.concatenate(data['p'], axis=1)[0:14,:,:]
        data['s'] = np.concatenate(data['s'], axis=1)[0:14,:,:]

        return data

    def save_sequence(self, data):
        file_name = "long_sequence_data.mat"
        savemat(os.path.join(self.config.tensor_board_dir, 'visual',
                             file_name), data)