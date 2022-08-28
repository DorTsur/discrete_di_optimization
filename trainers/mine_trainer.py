import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.6f}".format(x)})
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import os
from losses.mine_losses import DV_Loss, PMF_Loss, DV_Loss_regularized
from metrics.metrics import MINE_PMF_metrics
import matplotlib.pyplot as plt
from scipy import special as sp
from trainers.savers import PMFMINE_saver
import wandb


class CapEstMINE(object):
    def __init__(self, model, data, config):
        """
        MINE+PMF optimizer algorithm implementation
        """
        self.model = model
        self.data_iterator = data
        self.config = config
        if config.dv_regularize:
            self.dv_loss = DV_Loss_regularized(reg_coef=config.reg_coef)
        else:
            self.dv_loss = DV_Loss()
        self.pmf_loss = PMF_Loss(config)
        self.learning_rate = config.lr

        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.global_step_enc = tf.Variable(0, trainable=False, dtype=tf.int64)

        self.optimizer = {"dv": Adam(amsgrad=True, learning_rate=self.learning_rate),
                          "pmf": Adam(amsgrad=True,
                                      learning_rate=self.learning_rate/2)
                          }

        self.metrics = {"train": MINE_PMF_metrics(config.train_writer, name='dv_train'),
                        "eval": MINE_PMF_metrics(config.test_writer, name='dv_eval')}  # the trainer's metrics

        # self.noise_var = config.noise_var

        # if config.using_wandb:
        #     self.wandb_init()
        self.saver = PMFMINE_saver(config)
        self.snr = config.snr

        self.noise_var = 10 ** (-self.snr / 10)
        self.A = 1
        if config.constellation == '1d':
            if config.constellation_by_A:
                self.noise_var = 1
                self.A = np.sqrt(10 ** (self.snr / 10))
        if config.using_wandb:
            wandb.log({'noise_variance': self.noise_var})
            if config.constellation != '1d':
                wandb.log({'1D noise_variance': self.noise_var/2})

        self.gen_QAM_constellation()

    def sync_eval_model(self):
        # sync DV:
        w = self.model['dv'].get_weights()
        self.model['dv_eval'].set_weights(w)
        # sync pmf:
        w_pmf = self.model['pmf'].get_weights()  # similarly sync encoder model
        self.model['pmf_eval'].set_weights(w_pmf)

    def reset_model_states(self):
        def reset_recursively(models):
            for model in models.values():
                if isinstance(model, dict):
                    reset_recursively(model)
                else:
                    model.reset_states()

        reset_recursively(self.model)

    def train(self):
        for epoch in range(self.config.num_epochs):
            if epoch % self.config.eval_freq == 0:
                self.evaluate(epoch)  # perform evaluation step
                continue
            self.train_epoch(epoch)  # perform a training epoch

        self.evaluate(self.config.num_epochs, iterator="long_eval")  # perform a long evaluation

    def evaluate(self, epoch, iterator="eval"):
        self.sync_eval_model()
        self.metrics["eval"].reset_states()
        self.reset_model_states()
        # self.saver.reset_state()

        for input_batch in self.data_iterator[iterator]():
            output = self.eval_step(input_batch)
            self.metrics["eval"].update_state(output)

        if self.config.using_wandb:
            result = self.metrics["eval"].result()
            if iterator == "long_eval":
                wandb.log({'FINAL_MI_eval': result[0],
                           'PMF': output[2][0]})
            else:
                wandb.log({'Estimated_MI_eval': result[0],
                           'PMF': output[2][0]})

        self.metrics["eval"].log_metrics(epoch, model_name="Evaluation")
        self.saver.save(pmf=output[2][0].numpy(), model=self.model['pmf'], name=None if epoch > 0 else 'PMF.mat')
        self.constellation_vis(pmf=output[2][0].numpy(), dim=self.config.constellation)

    def DV_data(self, x, y):
        y_bar = tf.random.shuffle(y.numpy())
        input = tf.concat([x, y], axis=-1)
        input_bar = tf.concat([x, y_bar], axis=-1)

        return [input, input_bar]

    def channel(self, x):
        if self.config.constellation == '1d':
            return x + tf.cast(tf.random.normal(shape=tf.shape(x), stddev=tf.sqrt(self.noise_var)), dtype='float64')
        else:
            return x + tf.cast(tf.random.normal(shape=tf.shape(x), stddev=tf.sqrt(self.noise_var / 2)), dtype='float64')

    def eval_step(self, input_batch):
        logits = self.model['pmf'](input_batch, training=False)
        [ind, x] = tf.split(self.model['sampler'](logits), axis=-1, num_or_size_splits=[1, self.config.x_dim])
        y = self.channel(x)

        [input_xy, input_xy_bar] = self.DV_data(x, y)
        t = self.model['dv_eval'](input_xy, training=False)
        t_ = tf.exp(self.model['dv_eval'](input_xy_bar, training=False))

        norm_factor = tf.reduce_mean(tf.reduce_sum(tf.square(x), axis=-1))
        snr = 10*np.log10(norm_factor/self.noise_var)
        if self.config.using_wandb:
            wandb.log({'snr': snr})
        # print('snr={}'.format(snr))

        return [t, t_, tf.nn.softmax(logits), ind]

    def train_epoch(self, epoch):
        self.metrics["train"].reset_states()
        self.reset_model_states()

        # if np.random.rand() > 0.5 or epoch < 25:  # train DINE
        # if not(epoch % 25 == 0) or epoch < 25:  # train MINE
        if not (epoch % self.config.pmf_train_freq == 0) or epoch < 25:  # train MINE
            training_model = "MINE"
            for input_batch in self.data_iterator["train"]():
                outputs = self.train_step(input_batch, model=training_model)
                self.metrics["train"].update_state(outputs)

        else:  # train Enc.
            training_model = "PMF"
            for input_batch in self.data_iterator["train"]():
                outputs = self.train_step(input_batch, model=training_model)
                self.metrics["train"].update_state(outputs)

        print("Epoch = {}, training model is {}, est_mi = {}".format(epoch, training_model,
                                                                                self.metrics["train"].result()[0]))
        if epoch % 15 == 0:
            print("Input PMF is {}".format(sp.softmax(outputs[2].numpy()[0])))


        if self.config.using_wandb:
            wandb.log({'Estimated_MI_train': self.metrics["train"].result()[0]})
            wandb.log({'Epoch': epoch})

    def train_step(self, input_batch, model):
        if model == "MINE":
            gradients, t, logits, ind = self.compute_mine_grads(input_batch)
            self.apply_dv_grads(gradients)
        else:
            gradients, t, logits, ind = self.compute_pmf_grads(input_batch)  # calculate gradients
            self.apply_pmf_grads(gradients)
        return [t[0], t[1], logits, ind]

    @tf.function
    def apply_dv_grads(self, gradients_dv):
        self.optimizer['dv'].apply_gradients(zip(gradients_dv, self.model['dv'].trainable_weights))

    @tf.function
    def apply_pmf_grads(self, gradients_enc):
        self.optimizer['pmf'].apply_gradients(zip(gradients_enc, self.model['pmf'].trainable_weights))

    def compute_mine_grads(self, input_batch):
        with tf.GradientTape(persistent=True) as tape:
            logits = self.model['pmf'](input_batch, training=False)
            [ind, x] = tf.split(self.model['sampler'](logits), axis=-1, num_or_size_splits=[1, self.config.x_dim])
            y = self.channel(x)
            [input_xy, input_xy_bar] = self.DV_data(x, y)
            t = self.model['dv'](input_xy, training=True)
            t_ = tf.exp(self.model['dv'](input_xy_bar, training=True))
            loss = self.dv_loss(t, t_)

        gradients = tape.gradient(loss, self.model['dv'].trainable_weights)

        gradients, grad_norm = tf.clip_by_global_norm(gradients, self.config.clip_grad_norm_dv) # normalize gradients

        with self.config.train_writer.as_default():  # update trainer metrics
            tf.summary.scalar("grad_norm_dv", grad_norm, self.global_step)
            tf.summary.scalar("dv_loss", loss, self.global_step)
            # tf.summary.histogram(name="p", data=tf.nn.softmax(logits), step=self.global_step, buckets=50)
            self.global_step.assign_add(1)

        # print(loss)

        return gradients, [t, t_], logits, ind

    def compute_pmf_grads(self, input_batch):
        with tf.GradientTape(persistent=True) as tape:
            logits = self.model['pmf'](input_batch, training=True)
            [ind, x] = tf.split(self.model['sampler'](logits), axis=-1, num_or_size_splits=[1, self.config.x_dim])
            y = self.channel(x)
            [input_xy, input_xy_bar] = self.DV_data(x, y)

            t = self.model['dv'](input_xy)
            t_ = tf.exp(self.model['dv'](input_xy_bar))

            lp = self.enc_lp(logits, ind)

            # loss = -tf.reduce_mean(lp * tf.stop_gradient(t - tf.reduce_mean(t)))
            loss = -tf.reduce_mean(lp * tf.stop_gradient(t - tf.reduce_mean(t))) + self.pmf_regularizer(logits)
            # loss = -tf.reduce_mean(lp * tf.stop_gradient(t)) + self.pmf_regularizer(logits)


        gradients = tape.gradient(loss, self.model['pmf'].trainable_weights)

        gradients, grad_norm = tf.clip_by_global_norm(gradients, self.config.clip_grad_norm_pmf)  # normalize gradients

        with self.config.train_writer.as_default():  # update trainer metrics
            tf.summary.scalar("grad_norm_pmf", grad_norm, self.global_step)
            self.global_step.assign_add(1)

        return gradients, [t, t_], logits, ind

    def enc_lp(self, logits, ind):
        p = tf.nn.softmax(logits)
        n_ind = tf.cast(tf.expand_dims(tf.linspace(start=0, stop=self.config.batch_size - 1, num=self.config.batch_size), axis=-1),
                        'int64')
        x_ind = tf.concat([n_ind, tf.cast(ind, 'int64')], axis=-1)
        p_out = tf.math.log(tf.gather_nd(indices=x_ind, params=p))

        return tf.expand_dims(p_out, axis=-1)

    def wandb_init(self):
        self.run = wandb.init(project=self.config.wandb_project_name,
                              entity="dortsur",
                              config=self.config)
        self.wandb_config = wandb.config

    def gen_QAM_constellation(self):
        self.k = np.sqrt(self.config.x_alphabet)
        axe = np.linspace(start=-1, stop=1, num=int(self.k))
        Q0, Q1 = np.meshgrid(axe, axe)
        self.QAM_mat = tf.cast(np.stack([Q0.flatten(), Q1.flatten()], axis=-1), 'float64')
        self.qam_norm = tf.squeeze(tf.reduce_mean(tf.reduce_sum(tf.square(self.QAM_mat), axis=-1)))

    def pmf_regularizer(self,logits):
        p = tf.nn.softmax(logits[0, :])
        m = tf.reduce_sum(tf.square(self.QAM_mat), axis=-1)/self.qam_norm

        constellation_energy = tf.reduce_sum(m * p)


        # regularizer = tf.square(constellation_energy-1)

        regularizer = tf.square(constellation_energy)

        # def f1(): return tf.square(constellation_energy)
        # def f2(): return 0
        # regularizer = tf.cond(constellation_energy > 1, true_fn=f1, false_fn=f2)

        if self.config.using_wandb:
            wandb.log({'regularizer': regularizer})

        return self.config.pmf_regul_factor*regularizer

    def constellation_vis(self, pmf, dim):
        M = tf.shape(pmf)[0]
        if dim == '1d':
            constellation = np.linspace(-self.A, self.A, self.config.x_alphabet)
            fig = plt.figure()
            plt.bar(constellation, pmf, width=1 / M, color='darkslateblue', tick_label=[constellation[0]]+(M.numpy()-2)*['']+[constellation[-1]])
        else:
            constellation = self.QAM_mat/self.qam_norm
            markersize = 35 * tf.cast(M, 'float64') * pmf
            fig = plt.figure()
            plt.scatter(x=constellation[:, 0], y=constellation[:, 1], s=markersize, color='darkslateblue')
            plt.xlim((3 * np.amin(constellation[:, 0]), 3 * np.amax(constellation[:, 0])))
            plt.ylim((3 * np.amin(constellation[:, 1]), 3 * np.amax(constellation[:, 1])))

        path = os.path.join(self.config.tensor_board_dir, 'visual', 'constellation_pmf.png')
        plt.savefig(path)
        if self.config.using_wandb:
            wandb.log({"constellation" : wandb.Image(data_or_path=path)})
