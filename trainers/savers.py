import numpy as np

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
import matplotlib.pyplot as plt
import tensorflow as tf
import logging
import os
from scipy.io import savemat

logger = logging.getLogger("logger")


# noinspection PyCallingNonCallable
class Visualizer(object):
    def __init__(self, config):
        self.config = config
        self.save_path = os.path.join(config.tensor_board_dir, 'visual')

    def reset_state(self):
        pass

    def update_state(self, *args):
        pass

    def visualize(self):
        pass

    def save_raw_data(self):
        pass

# PARENT VISUALIZER CLASS
class DVVisualizer(Visualizer):
    def __init__(self, config):
        # Class for saving DV potentials values
        super().__init__(config)
        self.t_y_list = list()
        self.t_xy_list = list()

    def reset_state(self):
        self.t_y_list = list()
        self.t_xy_list = list()

    def update_state(self, t_y, t_xy):
        self.t_y_list.append(t_y)
        self.t_xy_list.append(t_xy)

    def convert_lists_to_np(self):
        t_y = [y[0] for y in self.t_y_list]
        t_y = tf.concat(t_y, axis=1)
        t_y_ = [y[1] for y in self.t_y_list]
        t_y_ = tf.concat(t_y_, axis=1)

        t_xy = [xy[0] for xy in self.t_xy_list]
        t_xy = tf.concat(t_xy, axis=1)
        t_xy_ = [xy[1] for xy in self.t_xy_list]
        t_xy_ = tf.concat(t_xy_, axis=1)
        return t_y, t_y_, t_xy, t_xy_

    def save(self, name=None):
        t_y, t_y_, t_xy, t_xy_ = self.convert_lists_to_np()

        file_name = name if name is not None else 'raw_data_latest.mat'
        savemat(os.path.join(self.config.tensor_board_dir, 'visual',
                             file_name), {"t_y": t_y.numpy(),
                                          "t_xy": t_xy.numpy()})

    def histogram(self,x):
        return

# USED BY BOTH CAPEST AND CAPESTCHKPT - RENAME
class DVEncoderVisualizer_perm1(DVVisualizer):
    def __init__(self, config):
        super().__init__(config)
        self.x_list = list()
        self.y_list = list()
        self.p_list = list()
        self.states_list = list()

    def reset_state(self):
        super().reset_state()
        self.x_list = list()
        self.y_list = list()
        self.p_list = list()
        self.states_list = list()

    def update_state(self, t_y, t_xy, x, y, p, states):
        super().update_state(t_y, t_xy)
        self.x_list.append(x)
        self.y_list.append(y)
        self.p_list.append(p)
        self.states_list.append(states)

    def convert_lists_to_np(self):
        t_y, t_y_, t_xy, t_xy_ = super().convert_lists_to_np()
        x_n = [x for x in self.x_list]
        # x_n = [x[0] for x in self.x_list]
        x_np = tf.concat(x_n, axis=1)
        y_n = [y for y in self.y_list]
        y_np = tf.concat(y_n, axis=1)
        p_n = [p for p in self.p_list]
        p_np = tf.concat(p_n, axis=1)
        states_n = [states for states in self.states_list]
        states_np = tf.concat(states_n, axis=1)
        return t_y, t_y_, t_xy, t_xy_, x_np, y_np, p_np, states_np

    def save(self, name=None):

        t_y, t_y_, t_xy, t_xy_, x, y, p, states = self.convert_lists_to_np()

        file_name = name if name is not None else 'raw_data_latest.mat'
        savemat(os.path.join(self.config.tensor_board_dir, 'visual',
                             file_name), {"t_y": t_y.numpy(),
                                          "t_xy": t_xy.numpy(),
                                          "x": x.numpy(),
                                          "y": y.numpy(),
                                          "p": p.numpy(),
                                          "states": states.numpy()})

    # def save_models(self, models, path):
    #     def save_recursively(models, path):
    #         for model in models:
    #             if isinstance(models[model], dict):
    #                 save_recursively(models[model], path)
    #             else:
    #                 path = os.path.join(path, model)
    #                 models[model].save_weights(filepath=path)
    #     save_recursively(models, path)

    def save_models(self, models, path):
        def save_recursively(models, path):
            for model in models:
                if isinstance(models[model], dict):
                    save_recursively(models[model], path)
                else:
                    path = os.path.join(path, model, model)
                    if model == 'enc':
                        models[model].save(filepath=os.path.join(path,"enc_model"))
                        # models[model].save_weights(filepath=os.path.join(path, model + "weights_h5.h5"),save_format="h5")
                    models[model].save_weights(filepath=os.path.join(path, model, "weights_tf", "weights"), save_format="tf")

        save_recursively(models, path)

# USED FOR QGRAPH TRAINER
class Q_est_saver(Visualizer):
    def __init__(self, config):
        super().__init__(config)
        self.q_list = list()
        self.y_list = list()
        self.s_list = list()
        self.q_hist = Histogram2d(name="Q_hist")

    def reset_state(self):
        self.q_list = list()
        self.s_list = list()
        self.y_list = list()

    def update_state(self, data):
        [q,s,y] = data
        self.q_list.append(q)
        self.s_list.append(s)
        self.y_list.append(y)
        self.q_hist.update_state(q)

    def convert_lists_to_np(self):
        s = tf.concat(self.s_list, axis=1)
        q = tf.concat(self.q_list, axis=1)
        y = tf.concat(self.y_list, axis=1)

        return s,q,y

    def save(self, name=None):
        s,q,y = self.convert_lists_to_np()
        self.q_hist.plot(save=True, save_path=os.path.join(self.config.tensor_board_dir, 'visual'),save_name="est_Q_hist.png")

        file_name = name if name is not None else 'raw_data_latest.mat'
        savemat(os.path.join(self.config.tensor_board_dir, 'visual',
                             file_name), {"s": s.numpy(),
                                          "q": q.numpy(),
                                          "y":y.numpy()})


    def save_models(self, models, path):
        def save_recursively(models, path):
            for model in models:
                if isinstance(models[model], dict):
                    save_recursively(models[model], path)
                else:
                    path = os.path.join(path, model)
                    models[model].save_weights(filepath=path)
        save_recursively(models, path)



####################
#  Figures classes
####################


class Figure(object):

    def __init__(self, name='fig', **kwargs):
        self.name = name
        self.fig_data = list()

    def reset_states(self):
        self.fig_data = list()

    def set_data(self, *args, **kwargs):
        pass

    def aggregate_data(self):
        if isinstance(self.fig_data, list):
            return np.concatenate(self.fig_data, axis=0)
        else:
            return self.fig_data

    def update_state(self, data):
        self.fig_data.append(data)

    def plot(self, save=None):
        pass

class Histogram2d(Figure):
    def __init__(self, name, **kwargs):
        super(Histogram2d, self).__init__(name, **kwargs)

    def aggregate_data(self):
        try:
            data = np.concatenate(self.fig_data, axis=1)
        except ValueError:
            return None
        return data
        # return np.reshape(data, [-1, data.shape[-1]])  # - ziv's line

    def plot(self, save=None, save_path="./visual", save_name="fig.png"):

        data = self.aggregate_data()

        if data is None:
            logger.info("no data aggregated at visualizer")
            return

        plt.figure()
        data_hist = np.reshape(data, newshape=[np.prod(data.shape[:-1]),data.shape[-1]])
        d = plt.hist2d(data_hist[100:, 0], data_hist[100:, 1], bins=50)
        plt.title(self.name)
        bins = d[0]
        edges = d[1]
        if save:
            plt.savefig(os.path.join(save_path, save_name))
            savemat(os.path.join(save_path, self.name + '_raw_data.mat'),
                    {"bins": bins,
                     "edges": edges,
                     "data":data})

        plt.close()

class Histogram(Figure):
    def __init__(self, name, **kwargs):
        super(Histogram, self).__init__(name, **kwargs)

    def aggregate_data(self):
        try:
            data = np.concatenate(self.fig_data, axis=0)
        except ValueError:
            return None
        return np.reshape(data, [-1, data.shape[-1]])

    def plot(self, save=None, save_path="./", save_name="fig.png"):

        data = self.aggregate_data()

        if data is None:
            logger.info("no data aggregated at visualizer")
            return

        plt.figure()
        d = plt.hist(data[100:], bins=np.linspace(np.min(data), np.max(data), 200))
        plt.title(self.name)
        bins = d[0]
        edges = d[1]
        if save:
            plt.savefig(os.path.join(save_path, save_name))
            savemat(os.path.join(save_path, self.name + '_raw_data.mat'),
                    {"bins": bins,
                     "edges": edges})
        plt.close()