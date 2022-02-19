import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
import logging
import os
from losses.dv_loss import Model_loss
from losses.enc_loss import Enc_Loss_Reinforce, Model_enc_loss
from metrics.metrics import ModelMetrics, ModelWithEncMetrics, PDINE_metrics, Q_est_metrics
from trainers.savers import DVEncoderVisualizer, DVEncoderVisualizer_states, DVEncoderVisualizer_perm1, Q_est_saver
from trainers.pdine_trainer import pdine_trainer
from trainers.dine_ndt_trainer import DI_Trainer_NDT
from data.Ising_FB_data_gen import Ising_Data, Ising, Ising_seq, IsingChannel_ziv, IsingChannel_state
from data.data_gens import Clean_Channel
from optimizers.lr import exp_dec_lr
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical
from scipy.io import savemat


logger = logging.getLogger("logger")

def build_trainer(model, data, config):
    if config.trainer_name == "di_with_enc_states":
        trainer = DI_Trainer_With_Encoder_states_no_p(model, data, config)  # DINE + Encoder (currently binary alphabets)
    elif config.trainer_name == "q_est":
        trainer = Q_est_trainer(model, data, config)
    elif config.trainer_name == "input_invest":
        trainer = InputDistInvestigator(model, data, config)
    elif config.trainer_name == "pdine_new":
        trainer = pdine_trainer(model, data, config)
    elif config.trainer_name == "cap_est_checkpoint":
        trainer = Cap_Est_Chkpt(model, data, config)
    else:
        raise ValueError("'{}' is an invalid trainer name")
    return trainer

    # if config.trainer_name == "di":
    #     trainer = DI_Trainer(model, data, config)  # only DINE
    # elif config.trainer_name == "di_with_enc":
    #     trainer = DI_Trainer_With_Encoder(model, data, config)  # DINE + Encoder (currently binary alphabets)
    # elif config.trainer_name == "di_with_enc_states":
    #     if config.with_p == 1:
    #         trainer = DI_Trainer_With_Encoder_states(model, data, config)  # DINE + Encoder (currently binary alphabets)
    #     else:
    #         trainer = DI_Trainer_With_Encoder_states_no_p(model, data, config)  # DINE + Encoder (currently binary alphabets)
    # elif config.trainer_name == "di_with_enc_states_new":
    #     trainer = DI_Trainer_With_Encoder_states_no_p_new(model, data, config)  # DINE + Encoder (currently binary alphabets)
    # elif config.trainer_name == "PDINE":
    #     trainer = PDINE_trainer(model, data, config)  # DINE + Encoder (currently binary alphabets)
    # elif config.trainer_name == "q_est":
    #     trainer = Q_est_trainer(model, data, config)
    # elif config.trainer_name == "input_invest":
    #     trainer = InputDistInvestigator(model, data, config)
    # elif config.trainer_name == "pdine_new":
    #     trainer = pdine_trainer(model, data, config)
    # elif config.trainer_name == "cont_ndt":
    #     trainer = DI_Trainer_NDT(model, data, config)
    # else:
    #     raise ValueError("'{}' is an invalid trainer name")
    # return trainer

######################
class DI_Trainer_With_Encoder_states_no_p(object):
    def __init__(self, model, data, config):
        """
        Constructor of trainer class, inherits from DINumericalTrainer
        :param model: a dictionary consisting of all models relevant to this trainer
        :param data: data loader instance, not used in this class.
        :param config: configuration parameters obtains for the relevant .json file
        """
        self.model = model  # the DINE model
        self.data_iterators = data  # the data generator
        self.config = config  # the configuration
        self.loss_fn = Model_loss()  # the DINE model loss function
        self.learning_rate = config.lr  # the model's learning rate
        if config.decay == 1:
            self.learning_rate_dv = exp_dec_lr(config, data, config.lr)
            self.learning_rate_enc = exp_dec_lr(config, data, config.lr/4)
        else:
            self.learning_rate_dv = config.lr
            self.learning_rate_enc = config.lr/4
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.bptt = config.bptt
        if config.optimizer == "adam":
            self.optimizer = {"dv_y": Adam(amsgrad=True, learning_rate=self.learning_rate_dv),
                              "dv_xy": Adam(amsgrad=True, learning_rate=self.learning_rate_dv),
                              "enc": Adam(amsgrad=True, learning_rate=self.learning_rate_enc)}  # the model's optimizers
        else:
            self.optimizer = {"dv_y": SGD(learning_rate=config.lr_SGD),
                              "dv_xy": SGD(learning_rate=config.lr_SGD),
                              "enc": SGD(learning_rate=config.lr_SGD/4)
                              }  # the model's optimizers
        if config.trainer_name == "di_with_enc_states":
            self.saver = DVEncoderVisualizer_perm1(config)
        else:
            self.saver = DVEncoderVisualizer(config)
        self.enc_loss_fn = Model_enc_loss(subtract=True)
        self.global_step_enc = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.metrics = {"train": ModelWithEncMetrics(config.train_writer, name='dv_train'),
                        "eval": ModelWithEncMetrics(config.test_writer, name='dv_eval')}  # the trainer's metrics
        self.feedback = (config.feedback == 1)
        self.T = config.T
        if self.feedback:
            self.fb_input = tf.zeros(shape=[config.batch_size, 1, config.x_dim + config.y_dim])
        else:
            self.fb_input = tf.zeros(shape=[config.batch_size, 1, config.x_dim])

    def train_epoch(self, epoch):
        self.metrics["train"].reset_states()
        self.reset_model_states()  # reset model and metrics states
        i = 0
        if np.random.rand() > 0.3 or epoch < 20:  # train DINE
            model_name = "DV epoch"
            for sample in self.data_iterators["train"]():
                # if sample is None:
                #     self.reset_model_states()  # in case we deal with a dataset
                #     continue
                # if i > 0:
                if i >= 0:
                    sample = self.fb_input
                output = self.train_dine_step(sample)  # calculate model outputs and perform a training step
                self.metrics["train"].update_state(output)  # update trainer metrics
        else:  # train Enc.
            model_name = "Encoder epoch"
            for sample in self.data_iterators["train"]():
                if sample is None:
                    self.reset_model_states()  # in case we deal with a dataset
                    continue
                output = self.train_enc_step(self.fb_input)  # calculate model outputs and perform a training step
                self.metrics["train"].update_state(output)  # update trainer metrics

        self.metrics["train"].log_metrics(epoch, model_name=model_name)  # log updated metrics

    def train_dine_step(self, sample):
        gradients_dv_y, gradients_dv_xy, t_y, t_xy, p = self.compute_dine_grads(sample)  # calculate gradients
        self.apply_grads(gradients_dv_y, gradients_dv_xy)  # apply gradients
        return [t_y, t_xy, p]

    def train_enc_step(self, sample):
        # gradients_enc, t_y, t_xy, p = self.compute_enc_grads(sample)  # calculate gradients
        gradients_enc, t_y, t_xy, p = self.compute_enc_grads_reinforce(sample)  # calculate gradients
        self.apply_enc_grads(gradients_enc)  # apply gradients
        return [t_y, t_xy, p]

    # @tf.function
    def compute_dine_grads(self, sample):  # gradients computation method
        with tf.GradientTape(persistent=True) as tape:
            [x, y, p] = self.model['enc'](sample, training=False)  # obtain samples through model
            # ## for debug:
            # p = 0.7 * tf.ones(shape=[self.config.batch_size, self.config.bptt, 1])
            # x, y = self.model['sampler'](p, training=False)
            # ##
            xy = tf.concat([x, y], axis=-1)  # t_xy model input
            t_y = self.model['dv']['y'](y, training=True)
            t_xy = self.model['dv']['xy'](xy, training=True)  # calculate models outputs
            loss = self.loss_fn(t_y, t_xy)  # calculate loss for each model
            loss_y, loss_xy = tf.split(loss, num_or_size_splits=2, axis=0)

        gradients_dv_y = tape.gradient(loss_y, self.model['dv']['y'].trainable_weights)
        gradients_dv_xy = tape.gradient(loss_xy, self.model['dv']['xy'].trainable_weights)  # calculate gradients

        gradients_dv_y, grad_norm_dv_y = tf.clip_by_global_norm(gradients_dv_y, self.config.clip_grad_norm)
        gradients_dv_xy, grad_norm_dv_xy = tf.clip_by_global_norm(gradients_dv_xy, self.config.clip_grad_norm)  # normalize gradients

        with self.config.train_writer.as_default():  # update trainer metrics
            tf.summary.scalar("grad_norm_dv_y", grad_norm_dv_y, self.global_step)
            tf.summary.scalar("grad_norm_dv_xy", grad_norm_dv_xy, self.global_step)
            tf.summary.scalar("x_mean", tf.math.reduce_mean(x), self.global_step)
            tf.summary.scalar("y_mean", tf.math.reduce_mean(y), self.global_step)
            self.global_step.assign_add(1)

        if self.feedback:
            x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
            y_ = tf.split(y, axis=1, num_or_size_splits=self.config.bptt)[-1]
            self.fb_input = tf.concat([x_, y_], axis=-1)
        else:
            x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
            self.fb_input = x_


        return gradients_dv_y, gradients_dv_xy, t_y, t_xy, p

    # @tf.function
    def compute_enc_grads(self, sample):  # gradients computation method
        with tf.GradientTape(persistent=True) as tape:
            [x, y, p] = self.model['enc'](sample, training=True)  # obtain samples through model
            xy = tf.concat([x, y], axis=-1)  # t_xy model input
            t_y = self.model['dv']['y'](y, training=False)
            t_xy = self.model['dv']['xy'](xy, training=False)  # calculate models outputs
            t_y_enc = self.calc_enc_loss_input(t_y, x, p)
            t_xy_enc = self.calc_enc_loss_input(t_xy, x, p)
            loss = self.enc_loss_fn(t_y_enc, t_xy_enc)

        gradients_enc = tape.gradient(loss, self.model['enc'].trainable_weights)

        gradients_enc, grad_norm_enc = tf.clip_by_global_norm(gradients_enc, self.config.clip_grad_norm)

        with self.config.train_writer.as_default():  # update trainer metrics
            tf.summary.scalar("grad_norm_enc", grad_norm_enc, self.global_step)
            self.global_step.assign_add(1)

        if self.feedback:
            x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
            y_ = tf.split(y, axis=1, num_or_size_splits=self.config.bptt)[-1]
            self.fb_input = tf.concat([x_, y_], axis=-1)
        else:
            x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
            self.fb_input = x_

        return gradients_enc, t_y, t_xy, p

    def compute_enc_grads_reinforce(self, sample):  # gradients computation method
        with tf.GradientTape(persistent=True) as tape:
            [x, y, p] = self.model['enc'](sample, training=True)  # obtain samples through model
            xy = tf.concat([x, y], axis=-1)  # t_xy model input
            t_y = self.model['dv']['y'](y, training=False)
            t_xy = self.model['dv']['xy'](xy, training=False)  # calculate models outputs
            #### reinforce idea ####
            lp = self.enc_lp(x,p)
            r = t_xy[0] - t_y[0]
            r = tf.squeeze(r, axis=-1)
            rho = tf.reduce_mean(r)
            T = self.T
            sum_list = [tf.reduce_sum(tf.slice(r, [0,k,0], [tf.shape(r)[0],T,tf.shape(r)[-1]]), axis=1, keepdims=True)
                        for k in range(tf.shape(r)[1]-T)]  # [B,1,1]
            rt = tf.concat(sum_list, axis=1)   # [B,bptt-T,1]
            [lp,_] = tf.split(lp, axis=1, num_or_size_splits=[self.bptt-T,T])
            loss = -tf.reduce_mean(lp*(rt-rho))
            ##### reinforce idea #####

        gradients_enc = tape.gradient(loss, self.model['enc'].trainable_weights)

        gradients_enc, grad_norm_enc = tf.clip_by_global_norm(gradients_enc, self.config.clip_grad_norm_enc)

        with self.config.train_writer.as_default():  # update trainer metrics
            tf.summary.scalar("grad_norm_enc", grad_norm_enc, self.global_step)
            tf.summary.scalar("mdp_loss", loss, self.global_step)
            self.global_step.assign_add(1)

        if self.feedback:
            x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
            y_ = tf.split(y, axis=1, num_or_size_splits=self.config.bptt)[-1]
            self.fb_input = tf.concat([x_, y_], axis=-1)
        else:
            x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
            self.fb_input = x_

        return gradients_enc, t_y, t_xy, p

    # @tf.function
    def calc_enc_loss_input(self, t_l, x, p):
        p_et = tf.stack(self.config.contrastive_duplicates * [p], axis=2)
        p_bar_et = tf.ones_like(p_et) - p_et
        p_t = tf.expand_dims(p, axis=2)
        p_bar_t = tf.ones_like(p_t) - p_t  # obtain p_bar values
        [t, et] = t_l
        et_mean = tf.reduce_mean(et)

        t_0, t_1 = t * tf.math.log(p_bar_t), t * tf.math.log(p_t)  # create tensor to draw values from
        et_0, et_1 = et * tf.math.log(p_bar_et), et * tf.math.log(p_et)

        x_t = tf.expand_dims(x, axis=2)
        x_et = tf.stack(self.config.contrastive_duplicates * [x], axis=2)

        t_enc = tf.where(tf.equal(x_t, 1), t_1, t_0)  # choose according to binary criteria
        et_enc = tf.where(tf.equal(x_et, 1), et_1, et_0)

        return [t_enc, et_enc / et_mean]

    def enc_lp(self, x, p):
        # p_et = tf.stack(self.config.contrastive_duplicates * [p], axis=2)
        # p_bar_et = tf.ones_like(p_et) - p_et
        # p_t = tf.expand_dims(p, axis=2)
        # p_bar_t = tf.ones_like(p_t) - p_t  # obtain p_bar values
        # [t, et] = t_l
        # et_mean = tf.reduce_mean(et)
        #
        # t_0, t_1 = t * tf.math.log(p_bar_t), t * tf.math.log(p_t)  # create tensor to draw values from
        # et_0, et_1 = et * tf.math.log(p_bar_et), et * tf.math.log(p_et)
        #
        # x_t = tf.expand_dims(x, axis=2)
        # x_et = tf.stack(self.config.contrastive_duplicates * [x], axis=2)
        #
        # t_enc = tf.where(tf.equal(x_t, 1), t_1, t_0)  # choose according to binary criteria
        # et_enc = tf.where(tf.equal(x_et, 1), et_1, et_0)
        p_out = tf.math.log(tf.where(tf.equal(x,1), p, 1-p))
        return p_out

    @tf.function
    def apply_enc_grads(self, gradients_enc):
        self.optimizer['enc'].apply_gradients(zip(gradients_enc, self.model['enc'].trainable_weights))

    def sync_eval_model(self):
        # sync DV:
        w_y = self.model['dv']['y'].get_weights()
        w_xy = self.model['dv']['xy'].get_weights()
        self.model['dv_eval']['y'].set_weights(w_y)
        self.model['dv_eval']['xy'].set_weights(w_xy)
        # sync enc:
        w_enc = self.model['enc'].get_weights()  # similarly sync encoder model
        self.model['enc_eval'].set_weights(w_enc)

    def evaluate(self, epoch, iterator="eval"):
        self.sync_eval_model()
        self.metrics["eval"].reset_states()
        self.reset_model_states()
        self.saver.reset_state()
        i=0
        p_t = []
        t_y_t = []
        states = []

        for input_batch in self.data_iterators[iterator]():
            input_batch = self.fb_input
            output = self.eval_step(input_batch)
            p_t.append(output[2])
            t_y_t.append(output[0][0])
            states.append(output[3])
            self.metrics["eval"].update_state(output[:-1])

        self.metrics["eval"].log_metrics(epoch, model_name="Evaluation")
        p_t = tf.concat(p_t, axis=1)
        t_y_t = tf.concat(t_y_t, axis=1)
        states = tf.concat(states, axis=1)
        with self.config.test_writer.as_default():
            tf.summary.histogram(name="p_hist", data=p_t, step=self.global_step, buckets=20)
            tf.summary.histogram(name="t_y_hist", data=t_y_t, step=self.global_step, buckets=20)
            tf.summary.histogram(name="states_y_hist", data=states, step=self.global_step, buckets=20)

        self.saver.save(None if epoch > 0 else 'begin.mat')
        if self.config.trainer_name == "di_with_enc_states":
            self.saver.save_models(models=self.model, path=os.path.join(self.config.tensor_board_dir, "models_weights"))

    # @tf.function
    def eval_step(self, input_batch):
        [x, y, p] = self.model['enc'](input_batch, training=False)
        xy = tf.concat([x, y], axis=-1)
        t_y = self.model['dv_eval']['y'](y, training=False)
        t_xy = self.model['dv_eval']['xy'](xy, training=False)
        # addition with states (now output is [t,et,s]:
        states_y = t_y[2]
        t_y = [t_y[0], t_y[1]]
        t_xy = [t_xy[0], t_xy[1]]

        self.saver.update_state(t_y, t_xy, x, y, p, states_y)

        if self.feedback:
            x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
            y_ = tf.split(y, axis=1, num_or_size_splits=self.config.bptt)[-1]
            self.fb_input = tf.concat([x_, y_], axis=-1)
        else:
            x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
            self.fb_input = x_

        return t_y, t_xy, p, states_y

    def train(self):
        for epoch in range(self.config.num_epochs):
            if epoch % self.config.eval_freq == 0:
                self.evaluate(epoch)  # perform evaluation step
                continue
            self.train_epoch(epoch)  # perform a training epoch

        self.evaluate(self.config.num_epochs, iterator="long_eval")  # perform a long evaluation

    def reset_model_states(self):
        def reset_recursively(models):
            for model in models.values():
                if isinstance(model, dict):
                    reset_recursively(model)
                else:
                    model.reset_states()

        reset_recursively(self.model)
        if (self.config.channel_name == "ising") or (self.config.channel_name == "trapdoor"):
            self.model['enc'].layers[3].reset_states()
            self.model['enc_eval'].layers[3].reset_states()

    @tf.function
    def apply_grads(self, gradients_dv_y, gradients_dv_xy):
        self.optimizer['dv_y'].apply_gradients(zip(gradients_dv_y, self.model['dv']['y'].trainable_weights))
        self.optimizer['dv_xy'].apply_gradients(zip(gradients_dv_xy, self.model['dv']['xy'].trainable_weights))


class Cap_Est_Chkpt(object):
    def __init__(self, model, data, config):
        """
        Constructor of trainer class, inherits from DINumericalTrainer
        :param model: a dictionary consisting of all models relevant to this trainer
        :param data: data loader instance, not used in this class.
        :param config: configuration parameters obtains for the relevant .json file
        """
        self.model = model  # the DINE model
        self.data_iterators = data  # the data generator
        self.config = config  # the configuration
        self.load_enc_weights()

        self.loss_fn = Model_loss()  # the DINE model loss function
        self.learning_rate = config.lr  # the model's learning rate
        if config.decay == 1:
            self.learning_rate_dv = exp_dec_lr(config, data, config.lr)
            self.learning_rate_enc = exp_dec_lr(config, data, config.lr/4)
        else:
            self.learning_rate_dv = config.lr
            self.learning_rate_enc = config.lr/4
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.bptt = config.bptt
        if config.optimizer == "adam":
            self.optimizer = {"dv_y": Adam(amsgrad=True, learning_rate=self.learning_rate_dv),
                              "dv_xy": Adam(amsgrad=True, learning_rate=self.learning_rate_dv),
                              "enc": Adam(amsgrad=True, learning_rate=self.learning_rate_enc)}  # the model's optimizers
        else:
            self.optimizer = {"dv_y": SGD(learning_rate=config.lr_SGD),
                              "dv_xy": SGD(learning_rate=config.lr_SGD),
                              "enc": SGD(learning_rate=config.lr_SGD/4)
                              }  # the model's optimizers
        self.saver = DVEncoderVisualizer_perm1(config)
        # if config.trainer_name == "di_with_enc_states":
        #     self.saver = DVEncoderVisualizer_perm1(config)
        # else:
        #     self.saver = DVEncoderVisualizer(config)
        self.enc_loss_fn = Model_enc_loss(subtract=True)
        self.global_step_enc = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.metrics = {"train": ModelWithEncMetrics(config.train_writer, name='dv_train'),
                        "eval": ModelWithEncMetrics(config.test_writer, name='dv_eval')}  # the trainer's metrics
        self.feedback = (config.feedback == 1)
        self.T = config.T
        if self.feedback:
            self.fb_input = tf.zeros(shape=[config.batch_size, 1, config.x_dim + config.y_dim])
        else:
            self.fb_input = tf.zeros(shape=[config.batch_size, 1, config.x_dim])

    def load_enc_weights(self):
        """
        model to set weights is encoder
        """
        # if self.channel_name == 'ising':
        #     p_val = self.config.p_ising
        # elif self.channel_name == 'trapdoor':
        #     p_val = self.config.p_trapdoor
        # elif self.channel_name == 'post':
        #     p_val = self.config.p_post
        # else:
        #     p_val = 0.5
        # dir_name = "enc_" + self.config.channel_name + str(p_val)
        # dir_name = os.path.join("pretrained_models_weights","enc_ising_01/enc.index")
        # weights_path = os.path.join(os.getcwd(),dir_name)
        # self.model['enc'].load_weights(weights_path)
        # for loading model:
        # loading model:
        # self.model['enc'] = None
        # filepath = os.path.join("./pretrained_models",self.config.channel_name+"_fb","p_05_4037_47k")
        # self.model['enc'] = tf.keras.models.load_model(filepath=filepath)

    #     loading weights:
    #     filepath = os.path.join("./pretrained_models", self.config.channel_name + "_fb", "p_02_emp_","weights")
        filepath = os.path.join("./pretrained_models", self.config.channel_name + "_ff", "1", "weights")
        self.model['enc'].load_weights(filepath)

    def train_epoch(self, epoch):
        self.metrics["train"].reset_states()
        self.reset_model_states()  # reset model and metrics states
        i = 0
        if np.random.rand() > 0.3 or epoch < 20:  # train DINE
            model_name = "DV epoch"
            for sample in self.data_iterators["train"]():
                # if sample is None:
                #     self.reset_model_states()  # in case we deal with a dataset
                #     continue
                # if i > 0:
                if i >= 0:
                    sample = self.fb_input
                output = self.train_dine_step(sample)  # calculate model outputs and perform a training step
                self.metrics["train"].update_state(output)  # update trainer metrics
        else:  # train Enc.
            model_name = "Encoder epoch"
            for sample in self.data_iterators["train"]():
                if sample is None:
                    self.reset_model_states()  # in case we deal with a dataset
                    continue
                output = self.train_enc_step(self.fb_input)  # calculate model outputs and perform a training step
                self.metrics["train"].update_state(output)  # update trainer metrics

        self.metrics["train"].log_metrics(epoch, model_name=model_name)  # log updated metrics

    def train_dine_step(self, sample):
        gradients_dv_y, gradients_dv_xy, t_y, t_xy, p = self.compute_dine_grads(sample)  # calculate gradients
        self.apply_grads(gradients_dv_y, gradients_dv_xy)  # apply gradients
        return [t_y, t_xy, p]

    def train_enc_step(self, sample):
        # gradients_enc, t_y, t_xy, p = self.compute_enc_grads(sample)  # calculate gradients
        gradients_enc, t_y, t_xy, p = self.compute_enc_grads_reinforce(sample)  # calculate gradients
        self.apply_enc_grads(gradients_enc)  # apply gradients
        return [t_y, t_xy, p]

    # @tf.function
    def compute_dine_grads(self, sample):  # gradients computation method
        with tf.GradientTape(persistent=True) as tape:
            [x, y, p] = self.model['enc'](sample, training=False)  # obtain samples through model
            # ## for debug:
            # p = 0.7 * tf.ones(shape=[self.config.batch_size, self.config.bptt, 1])
            # x, y = self.model['sampler'](p, training=False)
            # ##
            xy = tf.concat([x, y], axis=-1)  # t_xy model input
            t_y = self.model['dv']['y'](y, training=True)
            t_xy = self.model['dv']['xy'](xy, training=True)  # calculate models outputs
            loss = self.loss_fn(t_y, t_xy)  # calculate loss for each model
            loss_y, loss_xy = tf.split(loss, num_or_size_splits=2, axis=0)

        gradients_dv_y = tape.gradient(loss_y, self.model['dv']['y'].trainable_weights)
        gradients_dv_xy = tape.gradient(loss_xy, self.model['dv']['xy'].trainable_weights)  # calculate gradients

        gradients_dv_y, grad_norm_dv_y = tf.clip_by_global_norm(gradients_dv_y, self.config.clip_grad_norm)
        gradients_dv_xy, grad_norm_dv_xy = tf.clip_by_global_norm(gradients_dv_xy, self.config.clip_grad_norm)  # normalize gradients

        with self.config.train_writer.as_default():  # update trainer metrics
            tf.summary.scalar("grad_norm_dv_y", grad_norm_dv_y, self.global_step)
            tf.summary.scalar("grad_norm_dv_xy", grad_norm_dv_xy, self.global_step)
            tf.summary.scalar("x_mean", tf.math.reduce_mean(x), self.global_step)
            tf.summary.scalar("y_mean", tf.math.reduce_mean(y), self.global_step)
            self.global_step.assign_add(1)

        if self.feedback:
            x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
            y_ = tf.split(y, axis=1, num_or_size_splits=self.config.bptt)[-1]
            self.fb_input = tf.concat([x_, y_], axis=-1)
        else:
            x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
            self.fb_input = x_


        return gradients_dv_y, gradients_dv_xy, t_y, t_xy, p

    # @tf.function
    def compute_enc_grads(self, sample):  # gradients computation method
        with tf.GradientTape(persistent=True) as tape:
            [x, y, p] = self.model['enc'](sample, training=True)  # obtain samples through model
            xy = tf.concat([x, y], axis=-1)  # t_xy model input
            t_y = self.model['dv']['y'](y, training=False)
            t_xy = self.model['dv']['xy'](xy, training=False)  # calculate models outputs
            t_y_enc = self.calc_enc_loss_input(t_y, x, p)
            t_xy_enc = self.calc_enc_loss_input(t_xy, x, p)
            loss = self.enc_loss_fn(t_y_enc, t_xy_enc)

        gradients_enc = tape.gradient(loss, self.model['enc'].trainable_weights)

        gradients_enc, grad_norm_enc = tf.clip_by_global_norm(gradients_enc, self.config.clip_grad_norm)

        with self.config.train_writer.as_default():  # update trainer metrics
            tf.summary.scalar("grad_norm_enc", grad_norm_enc, self.global_step)
            self.global_step.assign_add(1)

        if self.feedback:
            x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
            y_ = tf.split(y, axis=1, num_or_size_splits=self.config.bptt)[-1]
            self.fb_input = tf.concat([x_, y_], axis=-1)
        else:
            x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
            self.fb_input = x_

        return gradients_enc, t_y, t_xy, p

    def compute_enc_grads_reinforce(self, sample):  # gradients computation method
        with tf.GradientTape(persistent=True) as tape:
            [x, y, p] = self.model['enc'](sample, training=True)  # obtain samples through model
            xy = tf.concat([x, y], axis=-1)  # t_xy model input
            t_y = self.model['dv']['y'](y, training=False)
            t_xy = self.model['dv']['xy'](xy, training=False)  # calculate models outputs
            #### reinforce idea ####
            lp = self.enc_lp(x,p)
            r = t_xy[0] - t_y[0]
            r = tf.squeeze(r, axis=-1)
            rho = tf.reduce_mean(r)
            T = self.T
            sum_list = [tf.reduce_sum(tf.slice(r, [0,k,0], [tf.shape(r)[0],T,tf.shape(r)[-1]]), axis=1, keepdims=True)
                        for k in range(tf.shape(r)[1]-T)]  # [B,1,1]
            rt = tf.concat(sum_list, axis=1)   # [B,bptt-T,1]
            [lp,_] = tf.split(lp, axis=1, num_or_size_splits=[self.bptt-T,T])
            loss = -tf.reduce_mean(lp*(rt-rho))
            ##### reinforce idea #####

        gradients_enc = tape.gradient(loss, self.model['enc'].trainable_weights)

        gradients_enc, grad_norm_enc = tf.clip_by_global_norm(gradients_enc, self.config.clip_grad_norm_enc)

        with self.config.train_writer.as_default():  # update trainer metrics
            tf.summary.scalar("grad_norm_enc", grad_norm_enc, self.global_step)
            tf.summary.scalar("mdp_loss", loss, self.global_step)
            self.global_step.assign_add(1)

        if self.feedback:
            x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
            y_ = tf.split(y, axis=1, num_or_size_splits=self.config.bptt)[-1]
            self.fb_input = tf.concat([x_, y_], axis=-1)
        else:
            x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
            self.fb_input = x_

        return gradients_enc, t_y, t_xy, p

    # @tf.function
    def calc_enc_loss_input(self, t_l, x, p):
        p_et = tf.stack(self.config.contrastive_duplicates * [p], axis=2)
        p_bar_et = tf.ones_like(p_et) - p_et
        p_t = tf.expand_dims(p, axis=2)
        p_bar_t = tf.ones_like(p_t) - p_t  # obtain p_bar values
        [t, et] = t_l
        et_mean = tf.reduce_mean(et)

        t_0, t_1 = t * tf.math.log(p_bar_t), t * tf.math.log(p_t)  # create tensor to draw values from
        et_0, et_1 = et * tf.math.log(p_bar_et), et * tf.math.log(p_et)

        x_t = tf.expand_dims(x, axis=2)
        x_et = tf.stack(self.config.contrastive_duplicates * [x], axis=2)

        t_enc = tf.where(tf.equal(x_t, 1), t_1, t_0)  # choose according to binary criteria
        et_enc = tf.where(tf.equal(x_et, 1), et_1, et_0)

        return [t_enc, et_enc / et_mean]

    def enc_lp(self, x, p):
        # p_et = tf.stack(self.config.contrastive_duplicates * [p], axis=2)
        # p_bar_et = tf.ones_like(p_et) - p_et
        # p_t = tf.expand_dims(p, axis=2)
        # p_bar_t = tf.ones_like(p_t) - p_t  # obtain p_bar values
        # [t, et] = t_l
        # et_mean = tf.reduce_mean(et)
        #
        # t_0, t_1 = t * tf.math.log(p_bar_t), t * tf.math.log(p_t)  # create tensor to draw values from
        # et_0, et_1 = et * tf.math.log(p_bar_et), et * tf.math.log(p_et)
        #
        # x_t = tf.expand_dims(x, axis=2)
        # x_et = tf.stack(self.config.contrastive_duplicates * [x], axis=2)
        #
        # t_enc = tf.where(tf.equal(x_t, 1), t_1, t_0)  # choose according to binary criteria
        # et_enc = tf.where(tf.equal(x_et, 1), et_1, et_0)
        p_out = tf.math.log(tf.where(tf.equal(x,1), p, 1-p))
        return p_out

    @tf.function
    def apply_enc_grads(self, gradients_enc):
        self.optimizer['enc'].apply_gradients(zip(gradients_enc, self.model['enc'].trainable_weights))

    def sync_eval_model(self):
        # sync DV:
        w_y = self.model['dv']['y'].get_weights()
        w_xy = self.model['dv']['xy'].get_weights()
        self.model['dv_eval']['y'].set_weights(w_y)
        self.model['dv_eval']['xy'].set_weights(w_xy)
        # sync enc:
        w_enc = self.model['enc'].get_weights()  # similarly sync encoder model
        self.model['enc_eval'].set_weights(w_enc)

    def evaluate(self, epoch, iterator="eval"):
        self.sync_eval_model()
        self.metrics["eval"].reset_states()
        self.reset_model_states()
        self.saver.reset_state()
        i=0
        p_t = []
        t_y_t = []
        states = []

        for input_batch in self.data_iterators[iterator]():
            input_batch = self.fb_input
            output = self.eval_step(input_batch)
            p_t.append(output[2])
            t_y_t.append(output[0][0])
            states.append(output[3])
            self.metrics["eval"].update_state(output[:-1])

        self.metrics["eval"].log_metrics(epoch, model_name="Evaluation")
        p_t = tf.concat(p_t, axis=1)
        t_y_t = tf.concat(t_y_t, axis=1)
        states = tf.concat(states, axis=1)
        with self.config.test_writer.as_default():
            tf.summary.histogram(name="p_hist", data=p_t, step=self.global_step, buckets=20)
            tf.summary.histogram(name="t_y_hist", data=t_y_t, step=self.global_step, buckets=20)
            tf.summary.histogram(name="states_y_hist", data=states, step=self.global_step, buckets=20)

        self.saver.save(None if epoch > 0 else 'begin.mat')
        # if self.config.trainer_name == "di_with_enc_states":
        self.saver.save_models(models=self.model, path=os.path.join(self.config.tensor_board_dir, "models_weights"))

    # @tf.function
    def eval_step(self, input_batch):
        [x, y, p] = self.model['enc'](input_batch, training=False)
        xy = tf.concat([x, y], axis=-1)
        t_y = self.model['dv_eval']['y'](y, training=False)
        t_xy = self.model['dv_eval']['xy'](xy, training=False)
        # addition with states (now output is [t,et,s]:
        states_y = t_y[2]
        t_y = [t_y[0], t_y[1]]
        t_xy = [t_xy[0], t_xy[1]]

        self.saver.update_state(t_y, t_xy, x, y, p, states_y)

        if self.feedback:
            x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
            y_ = tf.split(y, axis=1, num_or_size_splits=self.config.bptt)[-1]
            self.fb_input = tf.concat([x_, y_], axis=-1)
        else:
            x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
            self.fb_input = x_

        return t_y, t_xy, p, states_y

    def train(self):
        for epoch in range(self.config.num_epochs):
            if epoch % self.config.eval_freq == 0:
                self.evaluate(epoch)  # perform evaluation step
                continue
            self.train_epoch(epoch)  # perform a training epoch

        self.evaluate(self.config.num_epochs, iterator="long_eval")  # perform a long evaluation

    def reset_model_states(self):
        def reset_recursively(models):
            for model in models.values():
                if isinstance(model, dict):
                    reset_recursively(model)
                else:
                    model.reset_states()

        reset_recursively(self.model)
        if (self.config.channel_name == "ising") or (self.config.channel_name == "trapdoor"):
            self.model['enc'].layers[3].reset_states()
            self.model['enc_eval'].layers[3].reset_states()

    @tf.function
    def apply_grads(self, gradients_dv_y, gradients_dv_xy):
        self.optimizer['dv_y'].apply_gradients(zip(gradients_dv_y, self.model['dv']['y'].trainable_weights))
        self.optimizer['dv_xy'].apply_gradients(zip(gradients_dv_xy, self.model['dv']['xy'].trainable_weights))


class Q_est_trainer(object):
    def __init__(self, model, data, config):
        self.config = config
        self.model = model
        self.channel_name = config.channel_name
        self.load_enc_weights()
        self.data_iterator = data
        self.loss_function = CategoricalCrossentropy()  # q-graph estimator loss function
        self.feedback = config.feedback

        # learning rate:
        self.learning_rate = config.lr  # the model's learning rate
        if config.decay == 1:
            self.learning_rate = exp_dec_lr(config, data, config.lr / 4)
        else:
            self.learning_rate = config.lr / 4

        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.bptt = config.bptt
        if config.optimizer == "adam":
            self.optimizer = {"q_graph": Adam(amsgrad=True, learning_rate=self.learning_rate)}
        else:
            self.optimizer = {"q_graph": SGD(learning_rate=config.lr_SGD / 4)}  # the model's optimizers

        self.saver = Q_est_saver(config)

        # metrics:
        self.global_step_enc = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.metrics = {"train": Q_est_metrics(config.train_writer, name='dv_train'),
                            "eval": Q_est_metrics(config.test_writer, name='dv_eval')}

        if self.feedback:
            self.fb_input = tf.zeros(shape=[config.batch_size, 1, config.x_dim + config.y_dim])
        else:
            self.fb_input = tf.zeros(shape=[config.batch_size, 1, config.x_dim])


        # external data generator:
        self.external_gen = False
        if self.external_gen:
            self.data_gen = IsingChannel_state(bptt=config.bptt, input_shape=[config.batch_size,1])
            self.data_gen._call_()

    def evaluate(self, epoch, iterator="eval"):
        self.sync_eval_model()
        self.metrics["eval"].reset_states()
        self.reset_model_states()
        self.saver.reset_state()
        s_t = []
        q_t = []

        for input_batch in self.data_iterator[iterator]():
            input_batch = self.fb_input
            output = self.eval_step(input_batch)  # output = [q_est,s_onehot]
            q_t.append(output[0])
            s_t.append(output[1])
            self.metrics["eval"].update_state(output)

        self.metrics["eval"].log_metrics(epoch)
        q_t = tf.concat(q_t, axis=1)
        s_t = tf.concat(s_t, axis=1)
        with self.config.test_writer.as_default():
            tf.summary.histogram(name="Q_0 hist", data=q_t[:-1,0], step=self.global_step, buckets=50)
            tf.summary.histogram(name="Q_1 hist", data=q_t[:-1,1], step=self.global_step, buckets=50)

        name=None if epoch > 0 else 'begin.mat'
        self.saver.save(name=name)
        self.saver.save_models(models=self.model, path=self.config.tensor_board_dir)

    def eval_step(self, input_batch):
        if self.external_gen:
            [x, y] = self.data_gen._gen_()
            s = x
        else:
            [x, y, p, s] = self.model['enc'](input_batch, training=False)  # obtain samples through model
        # [x, y, p, s] = self.model['enc'](input_batch, training=False)  # obtain samples through model
        q_est = self.model['q_graph_test'](y, training=True)
        s_onehot = to_categorical(s, num_classes=2)
        # self.saver.histogram(q_est)

        self.saver.update_state([q_est, s_onehot, y])

        x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
        y_ = tf.split(y, axis=1, num_or_size_splits=self.config.bptt)[-1]

        if self.feedback:
            self.fb_input = tf.concat([x_, y_], axis=-1)
        else:
            self.fb_input = x_

        return [q_est, s_onehot]

    def load_enc_weights(self):
        """
        model to set weights is encoder
        """
        # if self.channel_name == 'ising':
        #     p_val = self.config.p_ising
        # elif self.channel_name == 'trapdoor':
        #     p_val = self.config.p_trapdoor
        # elif self.channel_name == 'post':
        #     p_val = self.config.p_post
        # else:
        #     p_val = 0.5
        # dir_name = "enc_" + self.config.channel_name + str(p_val)
        # dir_name = os.path.join("pretrained_models_weights","enc_ising_01/enc.index")
        # weights_path = os.path.join(os.getcwd(),dir_name)
        # self.model['enc'].load_weights(weights_path)
        # for loading model:
        # loading model:
        # self.model['enc'] = None
        # filepath = os.path.join("./pretrained_models",self.config.channel_name+"_fb","p_05_4037_47k")
        # self.model['enc'] = tf.keras.models.load_model(filepath=filepath)

    #     loading weights:
    #     filepath = os.path.join("./pretrained_models", self.config.channel_name + "_fb", "p_02_emp_","weights")
        filepath = os.path.join("./pretrained_models", self.config.channel_name + "_ff", "1", "weights")
        self.model['enc'].load_weights(filepath)

    def train(self):
        for epoch in range(self.config.num_epochs):
            if epoch % self.config.eval_freq == 0:
                self.evaluate(epoch)  # perform evaluation step
                continue
            self.train_epoch(epoch)  # perform a training epoch

        self.evaluate(self.config.num_epochs, iterator="long_eval")  # perform a long evaluation

    def train_epoch(self, epoch):
        self.metrics["train"].reset_states()
        self.reset_model_states()  # reset model and metrics states
        for sample in self.data_iterator["train"]():
            sample = self.fb_input
            output = self.train_step(sample)  # calculate model outputs and perform a training step
            self.metrics["train"].update_state(output)  # update trainer metrics

        self.metrics["train"].log_metrics(epoch)  # log updated metrics

    def train_step(self, sample):
        gradients, data_metrics = self.compute_grads(sample)
        self.apply_grads(gradients)
        return data_metrics

    def compute_grads(self, sample):  # gradients computation method
        with tf.GradientTape(persistent=True) as tape:
            if self.external_gen:
                [x, y] = self.data_gen._gen_()
                s = x
            else:
                [x, y, p, s] = self.model['enc'](sample, training=False)  # obtain samples through model
            q_est = self.model['q_graph'](y, training=True)
            s_onehot = to_categorical(s, num_classes=2)
            loss = self.loss_function(s_onehot, q_est)

        gradients = tape.gradient(loss, self.model['q_graph'].trainable_weights)
        gradients, grad_norm = tf.clip_by_global_norm(gradients, self.config.clip_grad_norm)

        with self.config.train_writer.as_default():  # update trainer metrics
            tf.summary.scalar("grad_norm", grad_norm, self.global_step)
            tf.summary.scalar("QE_loss", loss, self.global_step)
            self.global_step.assign_add(1)

        x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
        y_ = tf.split(y, axis=1, num_or_size_splits=self.config.bptt)[-1]

        if self.feedback:
            self.fb_input = tf.concat([x_, y_], axis=-1)
        else:
            self.fb_input = x_

        return [gradients, [q_est, s_onehot]]

    def apply_grads(self, gradients):
        self.optimizer['q_graph'].apply_gradients(zip(gradients, self.model['q_graph'].trainable_weights))

    # def sync_eval_model(self):
    #     # sync model:
    #     w = self.model['q_graph'].get_weights()
    #     self.model['q_graph_eval'].set_weights(w)

    def reset_model_states(self):
        def reset_recursively(models):
            for model in models.values():
                if isinstance(model, dict):
                    reset_recursively(model)
                else:
                    model.reset_states()

        reset_recursively(self.model)
        # if (self.config.channel_name == "ising") or (self.config.channel_name == "trapdoor"):
        #     self.model['enc'].layers[3].reset_states()
        #     self.model['enc_eval'].layers[3].reset_states()

    def sync_eval_model(self):
        # sync DV model:
        w = self.model['q_graph'].get_weights()
        self.model['q_graph_test'].set_weights(w)


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


class DI_Trainer_With_Encoder(object):
    def __init__(self, model, data, config):
        """
        Constructor of trainer class, inherits from DINumericalTrainer
        :param model: a dictionary consisting of all models relevant to this trainer
        :param data: data loader instance, not used in this class.
        :param config: configuration parameters obtains for the relevant .json file
        """
        self.model = model  # the DINE model
        self.data_iterators = data  # the data generator
        self.config = config  # the configuration
        self.loss_fn = Model_loss()  # the DINE model loss function
        self.learning_rate = config.lr  # the model's learning rate
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.optimizer = {"dv_y": Adam(amsgrad=True, learning_rate=self.learning_rate),
                          "dv_xy": Adam(amsgrad=True, learning_rate=self.learning_rate),
                          "enc": Adam(amsgrad=True, learning_rate=self.learning_rate/2),
                          "enc_eval": Adam(learning_rate=self.learning_rate)}  # the model's optimizers
        self.saver = DVEncoderVisualizer(config)

        self.enc_loss_fn = Model_enc_loss(subtract=True)
        self.global_step_enc = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.metrics = {"train": ModelWithEncMetrics(config.train_writer, name='dv_train'),
                        "eval": ModelWithEncMetrics(config.test_writer, name='dv_eval')}  # the trainer's metrics
        self.feedback = (config.feedback == 1)
        if self.feedback:
            self.fb_input = tf.zeros(shape=[config.batch_size, 1, config.x_dim + config.y_dim + 1])
            # self.fb_input = tf.zeros(shape=[config.batch_size, 1, config.x_dim])
        else:
            self.fb_input = tf.zeros(shape=[config.batch_size, 1, config.x_dim+1])
            # self.fb_input = tf.zeros(shape=[config.batch_size, 1, config.x_dim+config.y_dim+1])

    def train_epoch(self, epoch):
        self.metrics["train"].reset_states()
        self.reset_model_states()  # reset model and metrics states
        i = 0
        if np.random.rand() > 0.3 or epoch < 10:  # train DINE
            model_name = "DV epoch"
            for sample in self.data_iterators["train"]():
                if sample is None:
                    self.reset_model_states()  # in case we deal with a dataset
                    continue
                # if i > 0:
                if i >= 0:
                        sample = self.fb_input
                output = self.train_dine_step(sample)  # calculate model outputs and perform a training step
                self.metrics["train"].update_state(output[0], output[1], output[2])  # update trainer metrics
        else:  # train Enc.
            model_name = "Encoder epoch"
            for sample in self.data_iterators["train"]():
                if sample is None:
                    self.reset_model_states()  # in case we deal with a dataset
                    continue
                output = self.train_enc_step(self.fb_input)  # calculate model outputs and perform a training step
                self.metrics["train"].update_state(output[0], output[1], output[2])  # update trainer metrics

        self.metrics["train"].log_metrics(epoch, model_name=model_name)  # log updated metrics

    def train_dine_step(self, sample):
        gradients_dv_y, gradients_dv_xy, t_y, t_xy, p = self.compute_dine_grads(sample)  # calculate gradients
        self.apply_grads(gradients_dv_y, gradients_dv_xy)  # apply gradients
        return [t_y, t_xy, p]

    def train_enc_step(self, sample):
        gradients_enc, t_y, t_xy, p = self.compute_enc_grads(sample)  # calculate gradients
        self.apply_enc_grads(gradients_enc)  # apply gradients
        return [t_y, t_xy, p]

    # @tf.function
    def compute_dine_grads(self, sample):  # gradients computation method
        with tf.GradientTape(persistent=True) as tape:
            [x, y, p] = self.model['enc'](sample, training=False)  # obtain samples through model
            # ## for debug:
            # p = 0.7 * tf.ones(shape=[self.config.batch_size, self.config.bptt, 1])
            # x, y = self.model['sampler'](p, training=False)
            # ##
            xy = tf.concat([x, y], axis=-1)  # t_xy model input
            t_y = self.model['dv']['y'](y, training=True)
            t_xy = self.model['dv']['xy'](xy, training=True)  # calculate models outputs
            loss = self.loss_fn(t_y, t_xy)  # calculate loss for each model
            loss_y, loss_xy = tf.split(loss, num_or_size_splits=2, axis=0)

        gradients_dv_y = tape.gradient(loss_y, self.model['dv']['y'].trainable_weights)
        gradients_dv_xy = tape.gradient(loss_xy, self.model['dv']['xy'].trainable_weights)  # calculate gradients

        gradients_dv_y, grad_norm_dv_y = tf.clip_by_global_norm(gradients_dv_y, self.config.clip_grad_norm)
        gradients_dv_xy, grad_norm_dv_xy = tf.clip_by_global_norm(gradients_dv_xy, self.config.clip_grad_norm)  # normalize gradients

        with self.config.train_writer.as_default():  # update trainer metrics
            tf.summary.scalar("grad_norm_dv_y", grad_norm_dv_y, self.global_step)
            tf.summary.scalar("grad_norm_dv_xy", grad_norm_dv_xy, self.global_step)
            tf.summary.scalar("x_mean", tf.math.reduce_mean(x), self.global_step)
            tf.summary.scalar("y_mean", tf.math.reduce_mean(y), self.global_step)
            self.global_step.assign_add(1)

        if self.feedback:
            x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
            y_ = tf.split(y, axis=1, num_or_size_splits=self.config.bptt)[-1]
            p_ = tf.split(p, axis=1, num_or_size_splits=self.config.bptt)[-1]
            self.fb_input = tf.concat([p_, x_, y_], axis=-1)
        else:
            x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
            p_ = tf.split(p, axis=1, num_or_size_splits=self.config.bptt)[-1]
            self.fb_input = tf.concat([p_, x_], axis=-1)


        return gradients_dv_y, gradients_dv_xy, t_y, t_xy, p

    # @tf.function
    def compute_enc_grads(self, sample):  # gradients computation method
        with tf.GradientTape(persistent=True) as tape:
            [x, y, p] = self.model['enc'](sample, training=True)  # obtain samples through model
            xy = tf.concat([x, y], axis=-1)  # t_xy model input
            t_y = self.model['dv']['y'](y, training=False)
            t_xy = self.model['dv']['xy'](xy, training=False)  # calculate models outputs
            t_y_enc = self.calc_enc_loss_input(t_y, x, p)
            t_xy_enc = self.calc_enc_loss_input(t_xy, x, p)
            loss = self.enc_loss_fn(t_y_enc, t_xy_enc)

        gradients_enc = tape.gradient(loss, self.model['enc'].trainable_weights)

        gradients_enc, grad_norm_enc = tf.clip_by_global_norm(gradients_enc, self.config.clip_grad_norm)

        with self.config.train_writer.as_default():  # update trainer metrics
            tf.summary.scalar("grad_norm_enc", grad_norm_enc, self.global_step)
            self.global_step.assign_add(1)

        if self.feedback:
            x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
            y_ = tf.split(y, axis=1, num_or_size_splits=self.config.bptt)[-1]
            p_ = tf.split(p, axis=1, num_or_size_splits=self.config.bptt)[-1]
            self.fb_input = tf.concat([p_, x_, y_], axis=-1)
        else:
            x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
            p_ = tf.split(p, axis=1, num_or_size_splits=self.config.bptt)[-1]
            self.fb_input = tf.concat([p_, x_], axis=-1)

        return gradients_enc, t_y, t_xy, p

    # @tf.function
    def calc_enc_loss_input(self, t_l, x, p):
        p_et = tf.stack(self.config.contrastive_duplicates * [p], axis=2)
        p_bar_et = tf.ones_like(p_et) - p_et
        p_t = tf.expand_dims(p, axis=2)
        p_bar_t = tf.ones_like(p_t) - p_t  # obtain p_bar values
        [t, et] = t_l
        et_mean = tf.reduce_mean(et)

        t_0, t_1 = t * tf.math.log(p_bar_t), t * tf.math.log(p_t)  # create tensor to draw values from
        et_0, et_1 = et * tf.math.log(p_bar_et), et * tf.math.log(p_et)

        x_t = tf.expand_dims(x, axis=2)
        x_et = tf.stack(self.config.contrastive_duplicates * [x], axis=2)

        t_enc = tf.where(tf.equal(x_t, 1), t_1, t_0)  # choose according to binary criteria
        et_enc = tf.where(tf.equal(x_et, 1), et_1, et_0)

        return [t_enc, et_enc / et_mean]

    @tf.function
    def apply_enc_grads(self, gradients_enc):
        self.optimizer['enc'].apply_gradients(zip(gradients_enc, self.model['enc'].trainable_weights))

    def sync_eval_model(self):
        # sync DV:
        w_y = self.model['dv']['y'].get_weights()
        w_xy = self.model['dv']['xy'].get_weights()
        self.model['dv_eval']['y'].set_weights(w_y)
        self.model['dv_eval']['xy'].set_weights(w_xy)
        # sync enc:
        w_enc = self.model['enc'].get_weights()  # similarly sync encoder model
        self.model['enc_eval'].set_weights(w_enc)

    def evaluate(self, epoch, iterator="eval"):
        self.sync_eval_model()
        self.metrics["eval"].reset_states()
        self.reset_model_states()
        self.saver.reset_state()
        i=0
        p_t = []
        t_y_t = []

        for input_batch in self.data_iterators[iterator]():
            input_batch = self.fb_input
            output = self.eval_step(input_batch)
            p_t.append(output[2])
            t_y_t.append(output[0][0])
            self.metrics["eval"].update_state(t_y=output[0], t_xy=output[1], p=output[2])

        self.metrics["eval"].log_metrics(epoch, model_name="Evaluation")
        p_t = tf.concat(p_t, axis=1)
        t_y_t = tf.concat(t_y_t, axis=1)
        with self.config.test_writer.as_default():
            tf.summary.histogram(name="p_hist", data=p_t, step=self.global_step, buckets=20)
            tf.summary.histogram(name="t_y_hist", data=t_y_t, step=self.global_step, buckets=20)

        self.saver.save(None if epoch > 0 else 'begin.mat')

    # @tf.function
    def eval_step(self, input_batch):
        [x, y, p] = self.model['enc'](input_batch, training=False)
        xy = tf.concat([x, y], axis=-1)
        t_y = self.model['dv_eval']['y'](y, training=False)
        t_xy = self.model['dv_eval']['xy'](xy, training=False)

        self.saver.update_state(t_y, t_xy, x, y, p)

        if self.feedback:
            x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
            y_ = tf.split(y, axis=1, num_or_size_splits=self.config.bptt)[-1]
            p_ = tf.split(p, axis=1, num_or_size_splits=self.config.bptt)[-1]
            self.fb_input = tf.concat([p_, x_, y_], axis=-1)
        else:
            x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
            p_ = tf.split(p, axis=1, num_or_size_splits=self.config.bptt)[-1]
            self.fb_input = tf.concat([p_, x_], axis=-1)

        return t_y, t_xy, p

    def train(self):
        for epoch in range(self.config.num_epochs):
            if epoch % self.config.eval_freq == 0:
                self.evaluate(epoch)  # perform evaluation step
                continue
            self.train_epoch(epoch)  # perform a training epoch

        self.evaluate(self.config.num_epochs, iterator="long_eval")  # perform a long evaluation

    def reset_model_states(self):
        def reset_recursively(models):
            for model in models.values():
                if isinstance(model, dict):
                    reset_recursively(model)
                else:
                    model.reset_states()

        reset_recursively(self.model)

    @tf.function
    def apply_grads(self, gradients_dv_y, gradients_dv_xy):
        self.optimizer['dv_y'].apply_gradients(zip(gradients_dv_y, self.model['dv']['y'].trainable_weights))
        self.optimizer['dv_xy'].apply_gradients(zip(gradients_dv_xy, self.model['dv']['xy'].trainable_weights))



# ######################
# class DI_Trainer(object):
#     def __init__(self, model, data, config):
#         self.model = model  # the DINE model
#         self.data_iterators = data  # the data generator
#         self.config = config  # the configuration
#         self.loss_fn = Model_loss()  # the DINE model loss function
#         self.learning_rate = config.lr  # the model's learning rate
#         self.optimizer = {"dv_y": Adam(learning_rate=self.learning_rate),
#                           "dv_xy": Adam(learning_rate=self.learning_rate)}  # the model's optimizers
#         self.metrics = {"train": ModelMetrics(config.train_writer, name='dv_train'),
#                         "eval": ModelMetrics(config.test_writer, name='dv_eval')}  # the trainer's metrics
#         self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
#         ###### for ising di estimation #######
#         # self.ising_data = Ising_Data(config)
#         # self.ising_data = Ising(config)
#         self.ising_data = IsingChannel_ziv(input_shape=[config.batch_size, 1])
#         self.ising_data._call_()
#         # self.ising_data = Ising_seq(config)
#         # self.ising_data = Clean_Channel(config)
#
#     @tf.function
#     def compute_grads(self, x, y):  # gradients computation method
#         with tf.GradientTape(persistent=True) as tape:
#             xy = tf.concat([x, y], axis=-1)  # t_xy model input
#             t_y = self.model['dv']['y'](y, training=True)
#             t_xy = self.model['dv']['xy'](xy, training=True)  # calculate models outputs
#             loss = self.loss_fn(t_y, t_xy)  # calculate loss for each model
#             loss_y, loss_xy = tf.split(loss, num_or_size_splits=2, axis=0)
#
#         gradients_dv_y = tape.gradient(loss_y, self.model['dv']['y'].trainable_weights)
#         gradients_dv_xy = tape.gradient(loss_xy, self.model['dv']['xy'].trainable_weights)  # calculate gradients
#
#         gradients_dv_y, grad_norm_dv_y = tf.clip_by_global_norm(gradients_dv_y, self.config.clip_grad_norm)
#         gradients_dv_xy, grad_norm_dv_xy = tf.clip_by_global_norm(gradients_dv_xy, self.config.clip_grad_norm)  # normalize gradients
#
#         with self.config.train_writer.as_default():  # update trainer metrics
#             tf.summary.scalar("loss_y", loss_y[0], self.global_step)
#             tf.summary.scalar("loss_xy", loss_xy[0], self.global_step)
#             tf.summary.scalar("grad_norm_dv_y", grad_norm_dv_y, self.global_step)
#             tf.summary.scalar("grad_norm_dv_xy", grad_norm_dv_xy, self.global_step)
#             self.global_step.assign_add(1)
#
#         return gradients_dv_y, gradients_dv_xy, t_y, t_xy
#
#     @tf.function
#     def apply_grads(self, gradients_dv_y, gradients_dv_xy):
#         self.optimizer['dv_y'].apply_gradients(zip(gradients_dv_y, self.model['dv']['y'].trainable_weights))
#         self.optimizer['dv_xy'].apply_gradients(zip(gradients_dv_xy, self.model['dv']['xy'].trainable_weights))
#
#     @tf.function
#     def train_dine_step(self, x, y):
#         gradients_dv_y, gradients_dv_xy, t_y, t_xy = self.compute_grads(x, y)  # calculate gradients
#         self.apply_grads(gradients_dv_y, gradients_dv_xy)  # apply gradients
#         return t_y, t_xy
#
#     @tf.function
#     def eval_step(self, x, y):
#         xy = tf.concat([x, y], axis=-1)  # prepare inputs
#         t_y = self.model['dv_eval']['y'](y, training=False)  # caulcate model outputs
#         t_xy = self.model['dv_eval']['xy'](xy, training=False)
#         return t_y, t_xy
#
#     @staticmethod
#     def parse_train_sample(sample):
#         return sample
#
#     @staticmethod
#     def parse_eval_sample(sample):
#         return sample
#
#     @staticmethod
#     def parse_output_for_metric(output, sample):
#         return output
#
#     @staticmethod
#     def parse_output_for_visual(output, sample):
#         return output
#
#     def reset_model_states(self):
#         def reset_recursively(models):
#             for model in models.values():
#                 if isinstance(model, dict):
#                     reset_recursively(model)
#                 else:
#                     model.reset_states()
#
#         reset_recursively(self.model)
#
#     def sync_eval_model(self):  # syncing evaluation model weights with training model's
#         w_y = self.model['dv']['y'].get_weights()
#         w_xy = self.model['dv']['xy'].get_weights()
#         self.model['dv_eval']['y'].set_weights(w_y)
#         self.model['dv_eval']['xy'].set_weights(w_xy)
#
#     def train_step(self, sample):
#         return self.train_dine_step(*sample)
#
#     def train_epoch(self, epoch):
#         self.metrics["train"].reset_states()
#         self.reset_model_states()  # reset model and metrics states
#
#         for sample in self.data_iterators["train"]():
#             if sample is None:
#                 self.reset_model_states()  # in case we deal with a dataset
#                 continue
#             #### for ising di estimation
#             # input = self.ising_data.gen_data()
#             input = self.ising_data._gen_()
#             output = self.train_step(input)
#             ####
#             # output = self.train_step(sample)  # calculate model outputs and perform a training step
#             self.metrics["train"].update_state(*output)  # update trainer metrics
#
#         self.metrics["train"].log_metrics(epoch, model_name="DV")  # log updated metrics
#
#     def evaluate(self, epoch, iterator="eval"):
#         self.sync_eval_model()  # sync models weights
#         self.metrics["eval"].reset_states()  # reset all states
#         self.reset_model_states()
#         t_y_t = []
#
#         for sample in self.data_iterators[iterator]():
#             if sample is None:
#                 self.reset_model_states()
#                 continue
#             #### for ising di estimation
#             # input = self.ising_data.gen_data()
#             input = self.ising_data._gen_()
#             output = self.train_step(input)
#             ####
#             # output = self.eval_step(*sample)  # calculate models outputs
#             t_y,t_xy = output
#             t_y_t.append(t_y[0])
#
#             self.metrics["eval"].update_state(*self.parse_output_for_metric(output, sample))  # update evaluation metrics
#
#         t_y_t = tf.concat(t_y_t,axis=1)
#         with self.config.test_writer.as_default():
#             tf.summary.histogram(name="t_y_hist", data=t_y_t, step=self.global_step, buckets=20)
#
#
#         self.metrics["eval"].log_metrics(epoch, model_name="DV")  # log metrics
#
#     def train(self):
#         for epoch in range(self.config.num_epochs):
#             if epoch % self.config.eval_freq == 0:
#                 self.evaluate(epoch)  # perform evaluation step
#                 continue
#             self.train_epoch(epoch)  # perform a training epoch
#
#         self.evaluate(self.config.num_epochs, iterator="long_eval")  # perform a long evaluation
#
#
# class DI_Trainer_With_Encoder(object):
#     def __init__(self, model, data, config):
#         """
#         Constructor of trainer class, inherits from DINumericalTrainer
#         :param model: a dictionary consisting of all models relevant to this trainer
#         :param data: data loader instance, not used in this class.
#         :param config: configuration parameters obtains for the relevant .json file
#         """
#         self.model = model  # the DINE model
#         self.data_iterators = data  # the data generator
#         self.config = config  # the configuration
#         self.loss_fn = Model_loss()  # the DINE model loss function
#         self.learning_rate = config.lr  # the model's learning rate
#         self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
#         self.optimizer = {"dv_y": Adam(amsgrad=True, learning_rate=self.learning_rate),
#                           "dv_xy": Adam(amsgrad=True, learning_rate=self.learning_rate),
#                           "enc": Adam(amsgrad=True, learning_rate=self.learning_rate/2),
#                           "enc_eval": Adam(learning_rate=self.learning_rate)}  # the model's optimizers
#         self.saver = DVEncoderVisualizer(config)
#
#         self.enc_loss_fn = Model_enc_loss(subtract=True)
#         self.global_step_enc = tf.Variable(0, trainable=False, dtype=tf.int64)
#         self.metrics = {"train": ModelWithEncMetrics(config.train_writer, name='dv_train'),
#                         "eval": ModelWithEncMetrics(config.test_writer, name='dv_eval')}  # the trainer's metrics
#         self.feedback = (config.feedback == 1)
#         if self.feedback:
#             self.fb_input = tf.zeros(shape=[config.batch_size, 1, config.x_dim + config.y_dim + 1])
#             # self.fb_input = tf.zeros(shape=[config.batch_size, 1, config.x_dim])
#         else:
#             self.fb_input = tf.zeros(shape=[config.batch_size, 1, config.x_dim+1])
#             # self.fb_input = tf.zeros(shape=[config.batch_size, 1, config.x_dim+config.y_dim+1])
#
#     def train_epoch(self, epoch):
#         self.metrics["train"].reset_states()
#         self.reset_model_states()  # reset model and metrics states
#         i = 0
#         if np.random.rand() > 0.3 or epoch < 10:  # train DINE
#             model_name = "DV epoch"
#             for sample in self.data_iterators["train"]():
#                 if sample is None:
#                     self.reset_model_states()  # in case we deal with a dataset
#                     continue
#                 # if i > 0:
#                 if i >= 0:
#                         sample = self.fb_input
#                 output = self.train_dine_step(sample)  # calculate model outputs and perform a training step
#                 self.metrics["train"].update_state(output[0], output[1], output[2])  # update trainer metrics
#         else:  # train Enc.
#             model_name = "Encoder epoch"
#             for sample in self.data_iterators["train"]():
#                 if sample is None:
#                     self.reset_model_states()  # in case we deal with a dataset
#                     continue
#                 output = self.train_enc_step(self.fb_input)  # calculate model outputs and perform a training step
#                 self.metrics["train"].update_state(output[0], output[1], output[2])  # update trainer metrics
#
#         self.metrics["train"].log_metrics(epoch, model_name=model_name)  # log updated metrics
#
#     def train_dine_step(self, sample):
#         gradients_dv_y, gradients_dv_xy, t_y, t_xy, p = self.compute_dine_grads(sample)  # calculate gradients
#         self.apply_grads(gradients_dv_y, gradients_dv_xy)  # apply gradients
#         return [t_y, t_xy, p]
#
#     def train_enc_step(self, sample):
#         gradients_enc, t_y, t_xy, p = self.compute_enc_grads(sample)  # calculate gradients
#         self.apply_enc_grads(gradients_enc)  # apply gradients
#         return [t_y, t_xy, p]
#
#     # @tf.function
#     def compute_dine_grads(self, sample):  # gradients computation method
#         with tf.GradientTape(persistent=True) as tape:
#             [x, y, p] = self.model['enc'](sample, training=False)  # obtain samples through model
#             # ## for debug:
#             # p = 0.7 * tf.ones(shape=[self.config.batch_size, self.config.bptt, 1])
#             # x, y = self.model['sampler'](p, training=False)
#             # ##
#             xy = tf.concat([x, y], axis=-1)  # t_xy model input
#             t_y = self.model['dv']['y'](y, training=True)
#             t_xy = self.model['dv']['xy'](xy, training=True)  # calculate models outputs
#             loss = self.loss_fn(t_y, t_xy)  # calculate loss for each model
#             loss_y, loss_xy = tf.split(loss, num_or_size_splits=2, axis=0)
#
#         gradients_dv_y = tape.gradient(loss_y, self.model['dv']['y'].trainable_weights)
#         gradients_dv_xy = tape.gradient(loss_xy, self.model['dv']['xy'].trainable_weights)  # calculate gradients
#
#         gradients_dv_y, grad_norm_dv_y = tf.clip_by_global_norm(gradients_dv_y, self.config.clip_grad_norm)
#         gradients_dv_xy, grad_norm_dv_xy = tf.clip_by_global_norm(gradients_dv_xy, self.config.clip_grad_norm)  # normalize gradients
#
#         with self.config.train_writer.as_default():  # update trainer metrics
#             tf.summary.scalar("grad_norm_dv_y", grad_norm_dv_y, self.global_step)
#             tf.summary.scalar("grad_norm_dv_xy", grad_norm_dv_xy, self.global_step)
#             tf.summary.scalar("x_mean", tf.math.reduce_mean(x), self.global_step)
#             tf.summary.scalar("y_mean", tf.math.reduce_mean(y), self.global_step)
#             self.global_step.assign_add(1)
#
#         if self.feedback:
#             x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             y_ = tf.split(y, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             p_ = tf.split(p, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             self.fb_input = tf.concat([p_, x_, y_], axis=-1)
#         else:
#             x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             p_ = tf.split(p, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             self.fb_input = tf.concat([p_, x_], axis=-1)
#
#
#         return gradients_dv_y, gradients_dv_xy, t_y, t_xy, p
#
#     # @tf.function
#     def compute_enc_grads(self, sample):  # gradients computation method
#         with tf.GradientTape(persistent=True) as tape:
#             [x, y, p] = self.model['enc'](sample, training=True)  # obtain samples through model
#             xy = tf.concat([x, y], axis=-1)  # t_xy model input
#             t_y = self.model['dv']['y'](y, training=False)
#             t_xy = self.model['dv']['xy'](xy, training=False)  # calculate models outputs
#             t_y_enc = self.calc_enc_loss_input(t_y, x, p)
#             t_xy_enc = self.calc_enc_loss_input(t_xy, x, p)
#             loss = self.enc_loss_fn(t_y_enc, t_xy_enc)
#
#         gradients_enc = tape.gradient(loss, self.model['enc'].trainable_weights)
#
#         gradients_enc, grad_norm_enc = tf.clip_by_global_norm(gradients_enc, self.config.clip_grad_norm)
#
#         with self.config.train_writer.as_default():  # update trainer metrics
#             tf.summary.scalar("grad_norm_enc", grad_norm_enc, self.global_step)
#             self.global_step.assign_add(1)
#
#         if self.feedback:
#             x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             y_ = tf.split(y, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             p_ = tf.split(p, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             self.fb_input = tf.concat([p_, x_, y_], axis=-1)
#         else:
#             x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             p_ = tf.split(p, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             self.fb_input = tf.concat([p_, x_], axis=-1)
#
#         return gradients_enc, t_y, t_xy, p
#
#     # @tf.function
#     def calc_enc_loss_input(self, t_l, x, p):
#         p_et = tf.stack(self.config.contrastive_duplicates * [p], axis=2)
#         p_bar_et = tf.ones_like(p_et) - p_et
#         p_t = tf.expand_dims(p, axis=2)
#         p_bar_t = tf.ones_like(p_t) - p_t  # obtain p_bar values
#         [t, et] = t_l
#         et_mean = tf.reduce_mean(et)
#
#         t_0, t_1 = t * tf.math.log(p_bar_t), t * tf.math.log(p_t)  # create tensor to draw values from
#         et_0, et_1 = et * tf.math.log(p_bar_et), et * tf.math.log(p_et)
#
#         x_t = tf.expand_dims(x, axis=2)
#         x_et = tf.stack(self.config.contrastive_duplicates * [x], axis=2)
#
#         t_enc = tf.where(tf.equal(x_t, 1), t_1, t_0)  # choose according to binary criteria
#         et_enc = tf.where(tf.equal(x_et, 1), et_1, et_0)
#
#         return [t_enc, et_enc / et_mean]
#
#     @tf.function
#     def apply_enc_grads(self, gradients_enc):
#         self.optimizer['enc'].apply_gradients(zip(gradients_enc, self.model['enc'].trainable_weights))
#
#     def sync_eval_model(self):
#         # sync DV:
#         w_y = self.model['dv']['y'].get_weights()
#         w_xy = self.model['dv']['xy'].get_weights()
#         self.model['dv_eval']['y'].set_weights(w_y)
#         self.model['dv_eval']['xy'].set_weights(w_xy)
#         # sync enc:
#         w_enc = self.model['enc'].get_weights()  # similarly sync encoder model
#         self.model['enc_eval'].set_weights(w_enc)
#
#     def evaluate(self, epoch, iterator="eval"):
#         self.sync_eval_model()
#         self.metrics["eval"].reset_states()
#         self.reset_model_states()
#         self.saver.reset_state()
#         i=0
#         p_t = []
#         t_y_t = []
#
#         for input_batch in self.data_iterators[iterator]():
#             input_batch = self.fb_input
#             output = self.eval_step(input_batch)
#             p_t.append(output[2])
#             t_y_t.append(output[0][0])
#             self.metrics["eval"].update_state(t_y=output[0], t_xy=output[1], p=output[2])
#
#         self.metrics["eval"].log_metrics(epoch, model_name="Evaluation")
#         p_t = tf.concat(p_t, axis=1)
#         t_y_t = tf.concat(t_y_t, axis=1)
#         with self.config.test_writer.as_default():
#             tf.summary.histogram(name="p_hist", data=p_t, step=self.global_step, buckets=20)
#             tf.summary.histogram(name="t_y_hist", data=t_y_t, step=self.global_step, buckets=20)
#
#         self.saver.save(None if epoch > 0 else 'begin.mat')
#
#     # @tf.function
#     def eval_step(self, input_batch):
#         [x, y, p] = self.model['enc'](input_batch, training=False)
#         xy = tf.concat([x, y], axis=-1)
#         t_y = self.model['dv_eval']['y'](y, training=False)
#         t_xy = self.model['dv_eval']['xy'](xy, training=False)
#
#         self.saver.update_state(t_y, t_xy, x, y, p)
#
#         if self.feedback:
#             x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             y_ = tf.split(y, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             p_ = tf.split(p, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             self.fb_input = tf.concat([p_, x_, y_], axis=-1)
#         else:
#             x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             p_ = tf.split(p, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             self.fb_input = tf.concat([p_, x_], axis=-1)
#
#         return t_y, t_xy, p
#
#     def train(self):
#         for epoch in range(self.config.num_epochs):
#             if epoch % self.config.eval_freq == 0:
#                 self.evaluate(epoch)  # perform evaluation step
#                 continue
#             self.train_epoch(epoch)  # perform a training epoch
#
#         self.evaluate(self.config.num_epochs, iterator="long_eval")  # perform a long evaluation
#
#     def reset_model_states(self):
#         def reset_recursively(models):
#             for model in models.values():
#                 if isinstance(model, dict):
#                     reset_recursively(model)
#                 else:
#                     model.reset_states()
#
#         reset_recursively(self.model)
#
#     @tf.function
#     def apply_grads(self, gradients_dv_y, gradients_dv_xy):
#         self.optimizer['dv_y'].apply_gradients(zip(gradients_dv_y, self.model['dv']['y'].trainable_weights))
#         self.optimizer['dv_xy'].apply_gradients(zip(gradients_dv_xy, self.model['dv']['xy'].trainable_weights))
#
#
# class DI_Trainer_With_Encoder_states(object):
#     def __init__(self, model, data, config):
#         """
#         Constructor of trainer class, inherits from DINumericalTrainer
#         :param model: a dictionary consisting of all models relevant to this trainer
#         :param data: data loader instance, not used in this class.
#         :param config: configuration parameters obtains for the relevant .json file
#         """
#         self.model = model  # the DINE model
#         self.data_iterators = data  # the data generator
#         self.config = config  # the configuration
#         self.loss_fn = Model_loss()  # the DINE model loss function
#         self.learning_rate = config.lr  # the model's learning rate
#         self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
#         self.optimizer = {"dv_y": Adam(amsgrad=True, learning_rate=self.learning_rate),
#                           "dv_xy": Adam(amsgrad=True, learning_rate=self.learning_rate),
#                           "enc": Adam(amsgrad=True, learning_rate=self.learning_rate/2),
#                           "enc_eval": Adam(learning_rate=self.learning_rate)}  # the model's optimizers
#         if config.trainer_name == "di_with_enc_states":
#             self.saver = DVEncoderVisualizer_states(config)
#         else:
#             self.saver = DVEncoderVisualizer(config)
#         self.enc_loss_fn = Model_enc_loss(subtract=True)
#         self.global_step_enc = tf.Variable(0, trainable=False, dtype=tf.int64)
#         self.metrics = {"train": ModelWithEncMetrics(config.train_writer, name='dv_train'),
#                         "eval": ModelWithEncMetrics(config.test_writer, name='dv_eval')}  # the trainer's metrics
#         self.feedback = (config.feedback == 1)
#         if self.feedback:
#             self.fb_input = tf.zeros(shape=[config.batch_size, 1, config.x_dim + config.y_dim + 1])
#             # self.fb_input = tf.zeros(shape=[config.batch_size, 1, config.x_dim])
#         else:
#             self.fb_input = tf.zeros(shape=[config.batch_size, 1, config.x_dim+1])
#             # self.fb_input = tf.zeros(shape=[config.batch_size, 1, config.x_dim+config.y_dim+1])
#
#     def train_epoch(self, epoch):
#         self.metrics["train"].reset_states()
#         self.reset_model_states()  # reset model and metrics states
#         i = 0
#         if np.random.rand() > 0.3 or epoch < 10:  # train DINE
#             model_name = "DV epoch"
#             for sample in self.data_iterators["train"]():
#                 if sample is None:
#                     self.reset_model_states()  # in case we deal with a dataset
#                     continue
#                 # if i > 0:
#                 if i >= 0:
#                         sample = self.fb_input
#                 output = self.train_dine_step(sample)  # calculate model outputs and perform a training step
#                 self.metrics["train"].update_state(output[0], output[1], output[2])  # update trainer metrics
#         else:  # train Enc.
#             model_name = "Encoder epoch"
#             for sample in self.data_iterators["train"]():
#                 if sample is None:
#                     self.reset_model_states()  # in case we deal with a dataset
#                     continue
#                 output = self.train_enc_step(self.fb_input)  # calculate model outputs and perform a training step
#                 self.metrics["train"].update_state(output[0], output[1], output[2])  # update trainer metrics
#
#         self.metrics["train"].log_metrics(epoch, model_name=model_name)  # log updated metrics
#
#     def train_dine_step(self, sample):
#         gradients_dv_y, gradients_dv_xy, t_y, t_xy, p = self.compute_dine_grads(sample)  # calculate gradients
#         self.apply_grads(gradients_dv_y, gradients_dv_xy)  # apply gradients
#         return [t_y, t_xy, p]
#
#     def train_enc_step(self, sample):
#         gradients_enc, t_y, t_xy, p = self.compute_enc_grads(sample)  # calculate gradients
#         self.apply_enc_grads(gradients_enc)  # apply gradients
#         return [t_y, t_xy, p]
#
#     # @tf.function
#     def compute_dine_grads(self, sample):  # gradients computation method
#         with tf.GradientTape(persistent=True) as tape:
#             [x, y, p] = self.model['enc'](sample, training=False)  # obtain samples through model
#             # ## for debug:
#             # p = 0.7 * tf.ones(shape=[self.config.batch_size, self.config.bptt, 1])
#             # x, y = self.model['sampler'](p, training=False)
#             # ##
#             xy = tf.concat([x, y], axis=-1)  # t_xy model input
#             t_y = self.model['dv']['y'](y, training=True)
#             t_xy = self.model['dv']['xy'](xy, training=True)  # calculate models outputs
#             loss = self.loss_fn(t_y, t_xy)  # calculate loss for each model
#             loss_y, loss_xy = tf.split(loss, num_or_size_splits=2, axis=0)
#
#         gradients_dv_y = tape.gradient(loss_y, self.model['dv']['y'].trainable_weights)
#         gradients_dv_xy = tape.gradient(loss_xy, self.model['dv']['xy'].trainable_weights)  # calculate gradients
#
#         gradients_dv_y, grad_norm_dv_y = tf.clip_by_global_norm(gradients_dv_y, self.config.clip_grad_norm)
#         gradients_dv_xy, grad_norm_dv_xy = tf.clip_by_global_norm(gradients_dv_xy, self.config.clip_grad_norm)  # normalize gradients
#
#         with self.config.train_writer.as_default():  # update trainer metrics
#             tf.summary.scalar("grad_norm_dv_y", grad_norm_dv_y, self.global_step)
#             tf.summary.scalar("grad_norm_dv_xy", grad_norm_dv_xy, self.global_step)
#             tf.summary.scalar("x_mean", tf.math.reduce_mean(x), self.global_step)
#             tf.summary.scalar("y_mean", tf.math.reduce_mean(y), self.global_step)
#             self.global_step.assign_add(1)
#
#         if self.feedback:
#             x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             y_ = tf.split(y, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             p_ = tf.split(p, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             self.fb_input = tf.concat([p_, x_, y_], axis=-1)
#         else:
#             x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             p_ = tf.split(p, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             self.fb_input = tf.concat([p_, x_], axis=-1)
#
#
#         return gradients_dv_y, gradients_dv_xy, t_y, t_xy, p
#
#     # @tf.function
#     def compute_enc_grads(self, sample):  # gradients computation method
#         with tf.GradientTape(persistent=True) as tape:
#             [x, y, p] = self.model['enc'](sample, training=True)  # obtain samples through model
#             xy = tf.concat([x, y], axis=-1)  # t_xy model input
#             t_y = self.model['dv']['y'](y, training=False)
#             t_xy = self.model['dv']['xy'](xy, training=False)  # calculate models outputs
#             t_y_enc = self.calc_enc_loss_input(t_y, x, p)
#             t_xy_enc = self.calc_enc_loss_input(t_xy, x, p)
#             loss = self.enc_loss_fn(t_y_enc, t_xy_enc)
#
#         gradients_enc = tape.gradient(loss, self.model['enc'].trainable_weights)
#
#         gradients_enc, grad_norm_enc = tf.clip_by_global_norm(gradients_enc, self.config.clip_grad_norm)
#
#         with self.config.train_writer.as_default():  # update trainer metrics
#             tf.summary.scalar("grad_norm_enc", grad_norm_enc, self.global_step)
#             self.global_step.assign_add(1)
#
#         if self.feedback:
#             x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             y_ = tf.split(y, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             p_ = tf.split(p, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             self.fb_input = tf.concat([p_, x_, y_], axis=-1)
#         else:
#             x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             p_ = tf.split(p, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             self.fb_input = tf.concat([p_, x_], axis=-1)
#
#         return gradients_enc, t_y, t_xy, p
#
#     # @tf.function
#     def calc_enc_loss_input(self, t_l, x, p):
#         p_et = tf.stack(self.config.contrastive_duplicates * [p], axis=2)
#         p_bar_et = tf.ones_like(p_et) - p_et
#         p_t = tf.expand_dims(p, axis=2)
#         p_bar_t = tf.ones_like(p_t) - p_t  # obtain p_bar values
#         [t, et] = t_l
#         et_mean = tf.reduce_mean(et)
#
#         t_0, t_1 = t * tf.math.log(p_bar_t), t * tf.math.log(p_t)  # create tensor to draw values from
#         et_0, et_1 = et * tf.math.log(p_bar_et), et * tf.math.log(p_et)
#
#         x_t = tf.expand_dims(x, axis=2)
#         x_et = tf.stack(self.config.contrastive_duplicates * [x], axis=2)
#
#         t_enc = tf.where(tf.equal(x_t, 1), t_1, t_0)  # choose according to binary criteria
#         et_enc = tf.where(tf.equal(x_et, 1), et_1, et_0)
#
#         return [t_enc, et_enc / et_mean]
#
#     @tf.function
#     def apply_enc_grads(self, gradients_enc):
#         self.optimizer['enc'].apply_gradients(zip(gradients_enc, self.model['enc'].trainable_weights))
#
#     def sync_eval_model(self):
#         # sync DV:
#         w_y = self.model['dv']['y'].get_weights()
#         w_xy = self.model['dv']['xy'].get_weights()
#         self.model['dv_eval']['y'].set_weights(w_y)
#         self.model['dv_eval']['xy'].set_weights(w_xy)
#         # sync enc:
#         w_enc = self.model['enc'].get_weights()  # similarly sync encoder model
#         self.model['enc_eval'].set_weights(w_enc)
#
#     def evaluate(self, epoch, iterator="eval"):
#         self.sync_eval_model()
#         self.metrics["eval"].reset_states()
#         self.reset_model_states()
#         self.saver.reset_state()
#         i=0
#         p_t = []
#         t_y_t = []
#         states = []
#
#         for input_batch in self.data_iterators[iterator]():
#             input_batch = self.fb_input
#             output = self.eval_step(input_batch)
#             p_t.append(output[2])
#             t_y_t.append(output[0][0])
#             states.append(output[3])
#             self.metrics["eval"].update_state(t_y=output[0], t_xy=output[1], p=output[2])
#
#         self.metrics["eval"].log_metrics(epoch, model_name="Evaluation")
#         p_t = tf.concat(p_t, axis=1)
#         t_y_t = tf.concat(t_y_t, axis=1)
#         states = tf.concat(states, axis=1)
#         with self.config.test_writer.as_default():
#             tf.summary.histogram(name="p_hist", data=p_t, step=self.global_step, buckets=20)
#             tf.summary.histogram(name="t_y_hist", data=t_y_t, step=self.global_step, buckets=20)
#             tf.summary.histogram(name="states_y_hist", data=states, step=self.global_step, buckets=20)
#
#         self.saver.save(None if epoch > 0 else 'begin.mat')
#
#     # @tf.function
#     def eval_step(self, input_batch):
#         [x, y, p] = self.model['enc'](input_batch, training=False)
#         xy = tf.concat([x, y], axis=-1)
#         t_y = self.model['dv_eval']['y'](y, training=False)
#         t_xy = self.model['dv_eval']['xy'](xy, training=False)
#         # addition with states (now output is [t,et,s]:
#         states_y = t_y[2]
#         t_y = [t_y[0], t_y[1]]
#         t_xy = [t_xy[0], t_xy[1]]
#
#         self.saver.update_state(t_y, t_xy, x, y, p, states_y)
#
#         if self.feedback:
#             x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             y_ = tf.split(y, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             p_ = tf.split(p, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             self.fb_input = tf.concat([p_, x_, y_], axis=-1)
#         else:
#             x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             p_ = tf.split(p, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             self.fb_input = tf.concat([p_, x_], axis=-1)
#
#         return t_y, t_xy, p, states_y
#
#     def train(self):
#         for epoch in range(self.config.num_epochs):
#             if epoch % self.config.eval_freq == 0:
#                 self.evaluate(epoch)  # perform evaluation step
#                 continue
#             self.train_epoch(epoch)  # perform a training epoch
#
#         self.evaluate(self.config.num_epochs, iterator="long_eval")  # perform a long evaluation
#
#     def reset_model_states(self):
#         def reset_recursively(models):
#             for model in models.values():
#                 if isinstance(model, dict):
#                     reset_recursively(model)
#                 else:
#                     model.reset_states()
#
#         reset_recursively(self.model)
#
#     @tf.function
#     def apply_grads(self, gradients_dv_y, gradients_dv_xy):
#         self.optimizer['dv_y'].apply_gradients(zip(gradients_dv_y, self.model['dv']['y'].trainable_weights))
#         self.optimizer['dv_xy'].apply_gradients(zip(gradients_dv_xy, self.model['dv']['xy'].trainable_weights))
#
#
# class DI_Trainer_With_Encoder_states_no_p(object):
#     def __init__(self, model, data, config):
#         """
#         Constructor of trainer class, inherits from DINumericalTrainer
#         :param model: a dictionary consisting of all models relevant to this trainer
#         :param data: data loader instance, not used in this class.
#         :param config: configuration parameters obtains for the relevant .json file
#         """
#         self.model = model  # the DINE model
#         self.data_iterators = data  # the data generator
#         self.config = config  # the configuration
#         self.loss_fn = Model_loss()  # the DINE model loss function
#         self.learning_rate = config.lr  # the model's learning rate
#         if config.decay == 1:
#             self.learning_rate_dv = exp_dec_lr(config, data, config.lr)
#             self.learning_rate_enc = exp_dec_lr(config, data, config.lr/4)
#         else:
#             self.learning_rate_dv = config.lr
#             self.learning_rate_enc = config.lr/4
#         self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
#         self.bptt = config.bptt
#         if config.optimizer == "adam":
#             self.optimizer = {"dv_y": Adam(amsgrad=True, learning_rate=self.learning_rate_dv),
#                               "dv_xy": Adam(amsgrad=True, learning_rate=self.learning_rate_dv),
#                               "enc": Adam(amsgrad=True, learning_rate=self.learning_rate_enc)}  # the model's optimizers
#         else:
#             self.optimizer = {"dv_y": SGD(learning_rate=config.lr_SGD),
#                               "dv_xy": SGD(learning_rate=config.lr_SGD),
#                               "enc": SGD(learning_rate=config.lr_SGD/4)
#                               }  # the model's optimizers
#         if config.trainer_name == "di_with_enc_states":
#             self.saver = DVEncoderVisualizer_perm1(config)
#         else:
#             self.saver = DVEncoderVisualizer(config)
#         self.enc_loss_fn = Model_enc_loss(subtract=True)
#         self.global_step_enc = tf.Variable(0, trainable=False, dtype=tf.int64)
#         self.metrics = {"train": ModelWithEncMetrics(config.train_writer, name='dv_train'),
#                         "eval": ModelWithEncMetrics(config.test_writer, name='dv_eval')}  # the trainer's metrics
#         self.feedback = (config.feedback == 1)
#         self.T = config.T
#         if self.feedback:
#             self.fb_input = tf.zeros(shape=[config.batch_size, 1, config.x_dim + config.y_dim])
#         else:
#             self.fb_input = tf.zeros(shape=[config.batch_size, 1, config.x_dim])
#
#     def train_epoch(self, epoch):
#         self.metrics["train"].reset_states()
#         self.reset_model_states()  # reset model and metrics states
#         i = 0
#         if np.random.rand() > 0.3 or epoch < 20:  # train DINE
#             model_name = "DV epoch"
#             for sample in self.data_iterators["train"]():
#                 # if sample is None:
#                 #     self.reset_model_states()  # in case we deal with a dataset
#                 #     continue
#                 # if i > 0:
#                 if i >= 0:
#                     sample = self.fb_input
#                 output = self.train_dine_step(sample)  # calculate model outputs and perform a training step
#                 self.metrics["train"].update_state(output)  # update trainer metrics
#         else:  # train Enc.
#             model_name = "Encoder epoch"
#             for sample in self.data_iterators["train"]():
#                 if sample is None:
#                     self.reset_model_states()  # in case we deal with a dataset
#                     continue
#                 output = self.train_enc_step(self.fb_input)  # calculate model outputs and perform a training step
#                 self.metrics["train"].update_state(output)  # update trainer metrics
#
#         self.metrics["train"].log_metrics(epoch, model_name=model_name)  # log updated metrics
#
#     def train_dine_step(self, sample):
#         gradients_dv_y, gradients_dv_xy, t_y, t_xy, p = self.compute_dine_grads(sample)  # calculate gradients
#         self.apply_grads(gradients_dv_y, gradients_dv_xy)  # apply gradients
#         return [t_y, t_xy, p]
#
#     def train_enc_step(self, sample):
#         # gradients_enc, t_y, t_xy, p = self.compute_enc_grads(sample)  # calculate gradients
#         gradients_enc, t_y, t_xy, p = self.compute_enc_grads_reinforce(sample)  # calculate gradients
#         self.apply_enc_grads(gradients_enc)  # apply gradients
#         return [t_y, t_xy, p]
#
#     # @tf.function
#     def compute_dine_grads(self, sample):  # gradients computation method
#         with tf.GradientTape(persistent=True) as tape:
#             [x, y, p] = self.model['enc'](sample, training=False)  # obtain samples through model
#             # ## for debug:
#             # p = 0.7 * tf.ones(shape=[self.config.batch_size, self.config.bptt, 1])
#             # x, y = self.model['sampler'](p, training=False)
#             # ##
#             xy = tf.concat([x, y], axis=-1)  # t_xy model input
#             t_y = self.model['dv']['y'](y, training=True)
#             t_xy = self.model['dv']['xy'](xy, training=True)  # calculate models outputs
#             loss = self.loss_fn(t_y, t_xy)  # calculate loss for each model
#             loss_y, loss_xy = tf.split(loss, num_or_size_splits=2, axis=0)
#
#         gradients_dv_y = tape.gradient(loss_y, self.model['dv']['y'].trainable_weights)
#         gradients_dv_xy = tape.gradient(loss_xy, self.model['dv']['xy'].trainable_weights)  # calculate gradients
#
#         gradients_dv_y, grad_norm_dv_y = tf.clip_by_global_norm(gradients_dv_y, self.config.clip_grad_norm)
#         gradients_dv_xy, grad_norm_dv_xy = tf.clip_by_global_norm(gradients_dv_xy, self.config.clip_grad_norm)  # normalize gradients
#
#         with self.config.train_writer.as_default():  # update trainer metrics
#             tf.summary.scalar("grad_norm_dv_y", grad_norm_dv_y, self.global_step)
#             tf.summary.scalar("grad_norm_dv_xy", grad_norm_dv_xy, self.global_step)
#             tf.summary.scalar("x_mean", tf.math.reduce_mean(x), self.global_step)
#             tf.summary.scalar("y_mean", tf.math.reduce_mean(y), self.global_step)
#             self.global_step.assign_add(1)
#
#         if self.feedback:
#             x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             y_ = tf.split(y, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             self.fb_input = tf.concat([x_, y_], axis=-1)
#         else:
#             x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             self.fb_input = x_
#
#
#         return gradients_dv_y, gradients_dv_xy, t_y, t_xy, p
#
#     # @tf.function
#     def compute_enc_grads(self, sample):  # gradients computation method
#         with tf.GradientTape(persistent=True) as tape:
#             [x, y, p] = self.model['enc'](sample, training=True)  # obtain samples through model
#             xy = tf.concat([x, y], axis=-1)  # t_xy model input
#             t_y = self.model['dv']['y'](y, training=False)
#             t_xy = self.model['dv']['xy'](xy, training=False)  # calculate models outputs
#             t_y_enc = self.calc_enc_loss_input(t_y, x, p)
#             t_xy_enc = self.calc_enc_loss_input(t_xy, x, p)
#             loss = self.enc_loss_fn(t_y_enc, t_xy_enc)
#
#         gradients_enc = tape.gradient(loss, self.model['enc'].trainable_weights)
#
#         gradients_enc, grad_norm_enc = tf.clip_by_global_norm(gradients_enc, self.config.clip_grad_norm)
#
#         with self.config.train_writer.as_default():  # update trainer metrics
#             tf.summary.scalar("grad_norm_enc", grad_norm_enc, self.global_step)
#             self.global_step.assign_add(1)
#
#         if self.feedback:
#             x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             y_ = tf.split(y, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             self.fb_input = tf.concat([x_, y_], axis=-1)
#         else:
#             x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             self.fb_input = x_
#
#         return gradients_enc, t_y, t_xy, p
#
#     def compute_enc_grads_reinforce(self, sample):  # gradients computation method
#         with tf.GradientTape(persistent=True) as tape:
#             [x, y, p] = self.model['enc'](sample, training=True)  # obtain samples through model
#             xy = tf.concat([x, y], axis=-1)  # t_xy model input
#             t_y = self.model['dv']['y'](y, training=False)
#             t_xy = self.model['dv']['xy'](xy, training=False)  # calculate models outputs
#             #### reinforce idea ####
#             lp = self.enc_lp(x,p)
#             r = t_xy[0] - t_y[0]
#             r = tf.squeeze(r, axis=-1)
#             rho = tf.reduce_mean(r)
#             T = self.T
#             sum_list = [tf.reduce_sum(tf.slice(r, [0,k,0], [tf.shape(r)[0],T,tf.shape(r)[-1]]), axis=1, keepdims=True)
#                         for k in range(tf.shape(r)[1]-T)]  # [B,1,1]
#             rt = tf.concat(sum_list, axis=1)   # [B,bptt-T,1]
#             [lp,_] = tf.split(lp, axis=1, num_or_size_splits=[self.bptt-T,T])
#             loss = -tf.reduce_mean(lp*(rt-rho))
#             ##### reinforce idea #####
#
#         gradients_enc = tape.gradient(loss, self.model['enc'].trainable_weights)
#
#         gradients_enc, grad_norm_enc = tf.clip_by_global_norm(gradients_enc, self.config.clip_grad_norm_enc)
#
#         with self.config.train_writer.as_default():  # update trainer metrics
#             tf.summary.scalar("grad_norm_enc", grad_norm_enc, self.global_step)
#             tf.summary.scalar("mdp_loss", loss, self.global_step)
#             self.global_step.assign_add(1)
#
#         if self.feedback:
#             x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             y_ = tf.split(y, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             self.fb_input = tf.concat([x_, y_], axis=-1)
#         else:
#             x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             self.fb_input = x_
#
#         return gradients_enc, t_y, t_xy, p
#
#     # @tf.function
#     def calc_enc_loss_input(self, t_l, x, p):
#         p_et = tf.stack(self.config.contrastive_duplicates * [p], axis=2)
#         p_bar_et = tf.ones_like(p_et) - p_et
#         p_t = tf.expand_dims(p, axis=2)
#         p_bar_t = tf.ones_like(p_t) - p_t  # obtain p_bar values
#         [t, et] = t_l
#         et_mean = tf.reduce_mean(et)
#
#         t_0, t_1 = t * tf.math.log(p_bar_t), t * tf.math.log(p_t)  # create tensor to draw values from
#         et_0, et_1 = et * tf.math.log(p_bar_et), et * tf.math.log(p_et)
#
#         x_t = tf.expand_dims(x, axis=2)
#         x_et = tf.stack(self.config.contrastive_duplicates * [x], axis=2)
#
#         t_enc = tf.where(tf.equal(x_t, 1), t_1, t_0)  # choose according to binary criteria
#         et_enc = tf.where(tf.equal(x_et, 1), et_1, et_0)
#
#         return [t_enc, et_enc / et_mean]
#
#     def enc_lp(self, x, p):
#         # p_et = tf.stack(self.config.contrastive_duplicates * [p], axis=2)
#         # p_bar_et = tf.ones_like(p_et) - p_et
#         # p_t = tf.expand_dims(p, axis=2)
#         # p_bar_t = tf.ones_like(p_t) - p_t  # obtain p_bar values
#         # [t, et] = t_l
#         # et_mean = tf.reduce_mean(et)
#         #
#         # t_0, t_1 = t * tf.math.log(p_bar_t), t * tf.math.log(p_t)  # create tensor to draw values from
#         # et_0, et_1 = et * tf.math.log(p_bar_et), et * tf.math.log(p_et)
#         #
#         # x_t = tf.expand_dims(x, axis=2)
#         # x_et = tf.stack(self.config.contrastive_duplicates * [x], axis=2)
#         #
#         # t_enc = tf.where(tf.equal(x_t, 1), t_1, t_0)  # choose according to binary criteria
#         # et_enc = tf.where(tf.equal(x_et, 1), et_1, et_0)
#         p_out = tf.math.log(tf.where(tf.equal(x,1), p, 1-p))
#         return p_out
#
#     @tf.function
#     def apply_enc_grads(self, gradients_enc):
#         self.optimizer['enc'].apply_gradients(zip(gradients_enc, self.model['enc'].trainable_weights))
#
#     def sync_eval_model(self):
#         # sync DV:
#         w_y = self.model['dv']['y'].get_weights()
#         w_xy = self.model['dv']['xy'].get_weights()
#         self.model['dv_eval']['y'].set_weights(w_y)
#         self.model['dv_eval']['xy'].set_weights(w_xy)
#         # sync enc:
#         w_enc = self.model['enc'].get_weights()  # similarly sync encoder model
#         self.model['enc_eval'].set_weights(w_enc)
#
#     def evaluate(self, epoch, iterator="eval"):
#         self.sync_eval_model()
#         self.metrics["eval"].reset_states()
#         self.reset_model_states()
#         self.saver.reset_state()
#         i=0
#         p_t = []
#         t_y_t = []
#         states = []
#
#         for input_batch in self.data_iterators[iterator]():
#             input_batch = self.fb_input
#             output = self.eval_step(input_batch)
#             p_t.append(output[2])
#             t_y_t.append(output[0][0])
#             states.append(output[3])
#             self.metrics["eval"].update_state(output[:-1])
#
#         self.metrics["eval"].log_metrics(epoch, model_name="Evaluation")
#         p_t = tf.concat(p_t, axis=1)
#         t_y_t = tf.concat(t_y_t, axis=1)
#         states = tf.concat(states, axis=1)
#         with self.config.test_writer.as_default():
#             tf.summary.histogram(name="p_hist", data=p_t, step=self.global_step, buckets=20)
#             tf.summary.histogram(name="t_y_hist", data=t_y_t, step=self.global_step, buckets=20)
#             tf.summary.histogram(name="states_y_hist", data=states, step=self.global_step, buckets=20)
#
#         self.saver.save(None if epoch > 0 else 'begin.mat')
#         if self.config.trainer_name == "di_with_enc_states":
#             self.saver.save_models(models=self.model, path=os.path.join(self.config.tensor_board_dir, "models_weights"))
#
#     # @tf.function
#     def eval_step(self, input_batch):
#         [x, y, p] = self.model['enc'](input_batch, training=False)
#         xy = tf.concat([x, y], axis=-1)
#         t_y = self.model['dv_eval']['y'](y, training=False)
#         t_xy = self.model['dv_eval']['xy'](xy, training=False)
#         # addition with states (now output is [t,et,s]:
#         states_y = t_y[2]
#         t_y = [t_y[0], t_y[1]]
#         t_xy = [t_xy[0], t_xy[1]]
#
#         self.saver.update_state(t_y, t_xy, x, y, p, states_y)
#
#         if self.feedback:
#             x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             y_ = tf.split(y, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             self.fb_input = tf.concat([x_, y_], axis=-1)
#         else:
#             x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             self.fb_input = x_
#
#         return t_y, t_xy, p, states_y
#
#     def train(self):
#         for epoch in range(self.config.num_epochs):
#             if epoch % self.config.eval_freq == 0:
#                 self.evaluate(epoch)  # perform evaluation step
#                 continue
#             self.train_epoch(epoch)  # perform a training epoch
#
#         self.evaluate(self.config.num_epochs, iterator="long_eval")  # perform a long evaluation
#
#     def reset_model_states(self):
#         def reset_recursively(models):
#             for model in models.values():
#                 if isinstance(model, dict):
#                     reset_recursively(model)
#                 else:
#                     model.reset_states()
#
#         reset_recursively(self.model)
#         if (self.config.channel_name == "ising") or (self.config.channel_name == "trapdoor"):
#             self.model['enc'].layers[3].reset_states()
#             self.model['enc_eval'].layers[3].reset_states()
#
#     @tf.function
#     def apply_grads(self, gradients_dv_y, gradients_dv_xy):
#         self.optimizer['dv_y'].apply_gradients(zip(gradients_dv_y, self.model['dv']['y'].trainable_weights))
#         self.optimizer['dv_xy'].apply_gradients(zip(gradients_dv_xy, self.model['dv']['xy'].trainable_weights))
#
#
# class DI_Trainer_With_Encoder_states_no_p_new(object):
#     def __init__(self, model, data, config):
#         """
#         Constructor of trainer class, inherits from DINumericalTrainer
#         :param model: a dictionary consisting of all models relevant to this trainer
#         :param data: data loader instance, not used in this class.
#         :param config: configuration parameters obtains for the relevant .json file
#         """
#         self.model = model  # the DINE model
#         self.data_iterators = data  # the data generator
#         self.config = config  # the configuration
#         self.loss_fn = Model_loss()  # the DINE model loss function
#         self.learning_rate = config.lr  # the model's learning rate
#         if config.decay == 1:
#             self.learning_rate_dv = exp_dec_lr(config, data, config.lr)
#             self.learning_rate_enc = exp_dec_lr(config, data, config.lr/4)
#         else:
#             self.learning_rate_dv = config.lr
#             self.learning_rate_enc = config.lr/4
#         self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
#         self.bptt = config.bptt
#         if config.optimizer == "adam":
#             self.optimizer = {"dv_y": Adam(amsgrad=True, learning_rate=self.learning_rate_dv),
#                               "dv_xy": Adam(amsgrad=True, learning_rate=self.learning_rate_dv),
#                               "enc": Adam(amsgrad=True, learning_rate=self.learning_rate_enc)}  # the model's optimizers
#         else:
#             self.optimizer = {"dv_y": SGD(learning_rate=config.lr_SGD),
#                               "dv_xy": SGD(learning_rate=config.lr_SGD),
#                               "enc": SGD(learning_rate=config.lr_SGD/4)
#                               }  # the model's optimizers
#         if config.trainer_name == "di_with_enc_states_new":
#             self.saver = DVEncoderVisualizer_perm1(config)
#         else:
#             self.saver = DVEncoderVisualizer(config)
#         self.enc_loss_fn = Model_enc_loss(subtract=True)
#         self.global_step_enc = tf.Variable(0, trainable=False, dtype=tf.int64)
#         self.metrics = {"train": ModelWithEncMetrics(config.train_writer, name='dv_train'),
#                         "eval": ModelWithEncMetrics(config.test_writer, name='dv_eval')}  # the trainer's metrics
#         self.feedback = (config.feedback == 1)
#         self.T = config.T
#         if self.feedback:
#             self.fb_input = tf.zeros(shape=[config.batch_size, 1, config.x_dim + config.y_dim])
#         else:
#             self.fb_input = tf.zeros(shape=[config.batch_size, 1, config.x_dim])
#
#     def train_epoch(self, epoch):
#         self.metrics["train"].reset_states()
#         self.reset_model_states()  # reset model and metrics states
#         i = 0
#         if np.random.rand() > 0.3 or epoch < 20:  # train DINE
#             model_name = "DV epoch"
#             for sample in self.data_iterators["train"]():
#                 # if sample is None:
#                 #     self.reset_model_states()  # in case we deal with a dataset
#                 #     continue
#                 # if i > 0:
#                 if i >= 0:
#                         sample = self.fb_input
#                 output = self.train_dine_step(sample)  # calculate model outputs and perform a training step
#                 self.metrics["train"].update_state(output)  # update trainer metrics
#         else:  # train Enc.
#             model_name = "Encoder epoch"
#             for sample in self.data_iterators["train"]():
#                 if sample is None:
#                     self.reset_model_states()  # in case we deal with a dataset
#                     continue
#                 output = self.train_enc_step(self.fb_input)  # calculate model outputs and perform a training step
#                 self.metrics["train"].update_state(output)  # update trainer metrics
#
#         self.metrics["train"].log_metrics(epoch, model_name=model_name)  # log updated metrics
#
#     def train_dine_step(self, sample):
#         gradients_dv_y, gradients_dv_xy, t_y, t_xy, p = self.compute_dine_grads(sample)  # calculate gradients
#         self.apply_grads(gradients_dv_y, gradients_dv_xy)  # apply gradients
#         return [t_y, t_xy, p]
#
#     def train_enc_step(self, sample):
#         # gradients_enc, t_y, t_xy, p = self.compute_enc_grads(sample)  # calculate gradients
#         gradients_enc, t_y, t_xy, p = self.compute_enc_grads_reinforce(sample)  # calculate gradients
#         self.apply_enc_grads(gradients_enc)  # apply gradients
#         return [t_y, t_xy, p]
#
#     # @tf.function
#     def compute_dine_grads(self, sample):  # gradients computation method
#         with tf.GradientTape(persistent=True) as tape:
#             [x, y, p, s] = self.model['enc'](sample, training=False)  # obtain samples through model
#             # ## for debug:
#             # p = 0.7 * tf.ones(shape=[self.config.batch_size, self.config.bptt, 1])
#             # x, y = self.model['sampler'](p, training=False)
#             # ##
#             xy = tf.concat([x, y], axis=-1)  # t_xy model input
#             t_y = self.model['dv']['y'](y, training=True)
#             t_xy = self.model['dv']['xy'](xy, training=True)  # calculate models outputs
#             loss = self.loss_fn(t_y, t_xy)  # calculate loss for each model
#             loss_y, loss_xy = tf.split(loss, num_or_size_splits=2, axis=0)
#
#         gradients_dv_y = tape.gradient(loss_y, self.model['dv']['y'].trainable_weights)
#         gradients_dv_xy = tape.gradient(loss_xy, self.model['dv']['xy'].trainable_weights)  # calculate gradients
#
#         gradients_dv_y, grad_norm_dv_y = tf.clip_by_global_norm(gradients_dv_y, self.config.clip_grad_norm)
#         gradients_dv_xy, grad_norm_dv_xy = tf.clip_by_global_norm(gradients_dv_xy, self.config.clip_grad_norm)  # normalize gradients
#
#         with self.config.train_writer.as_default():  # update trainer metrics
#             tf.summary.scalar("grad_norm_dv_y", grad_norm_dv_y, self.global_step)
#             tf.summary.scalar("grad_norm_dv_xy", grad_norm_dv_xy, self.global_step)
#             tf.summary.scalar("x_mean", tf.math.reduce_mean(x), self.global_step)
#             tf.summary.scalar("y_mean", tf.math.reduce_mean(y), self.global_step)
#             self.global_step.assign_add(1)
#
#         if self.feedback:
#             x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             y_ = tf.split(y, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             self.fb_input = tf.concat([x_, y_], axis=-1)
#         else:
#             x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             self.fb_input = x_
#
#
#         return gradients_dv_y, gradients_dv_xy, t_y, t_xy, p
#
#     # @tf.function
#     def compute_enc_grads(self, sample):  # gradients computation method
#         with tf.GradientTape(persistent=True) as tape:
#             [x, y, p, s] = self.model['enc'](sample, training=True)  # obtain samples through model
#             xy = tf.concat([x, y], axis=-1)  # t_xy model input
#             t_y = self.model['dv']['y'](y, training=False)
#             t_xy = self.model['dv']['xy'](xy, training=False)  # calculate models outputs
#             t_y_enc = self.calc_enc_loss_input(t_y, x, p)
#             t_xy_enc = self.calc_enc_loss_input(t_xy, x, p)
#             loss = self.enc_loss_fn(t_y_enc, t_xy_enc)
#
#         gradients_enc = tape.gradient(loss, self.model['enc'].trainable_weights)
#
#         gradients_enc, grad_norm_enc = tf.clip_by_global_norm(gradients_enc, self.config.clip_grad_norm)
#
#         with self.config.train_writer.as_default():  # update trainer metrics
#             tf.summary.scalar("grad_norm_enc", grad_norm_enc, self.global_step)
#             self.global_step.assign_add(1)
#
#         if self.feedback:
#             x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             y_ = tf.split(y, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             self.fb_input = tf.concat([x_, y_], axis=-1)
#         else:
#             x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             self.fb_input = x_
#
#         return gradients_enc, t_y, t_xy, p
#
#     def compute_enc_grads_reinforce(self, sample):  # gradients computation method
#         with tf.GradientTape(persistent=True) as tape:
#             [x, y, p, s] = self.model['enc'](sample, training=True)  # obtain samples through model
#             xy = tf.concat([x, y], axis=-1)  # t_xy model input
#             t_y = self.model['dv']['y'](y, training=False)
#             t_xy = self.model['dv']['xy'](xy, training=False)  # calculate models outputs
#             #### reinforce idea ####
#             lp = self.enc_lp(x,p)
#             r = t_xy[0] - t_y[0]
#             r = tf.squeeze(r, axis=-1)
#             rho = tf.reduce_mean(r)
#             T = self.T
#             sum_list = [tf.reduce_sum(tf.slice(r, [0,k,0], [tf.shape(r)[0],T,tf.shape(r)[-1]]), axis=1, keepdims=True)
#                         for k in range(tf.shape(r)[1]-T)]  # [B,1,1]
#             rt = tf.concat(sum_list, axis=1)   # [B,bptt-T,1]
#             [lp,_] = tf.split(lp, axis=1, num_or_size_splits=[self.bptt-T,T])
#             loss = -tf.reduce_mean(lp*(rt-rho))
#             ##### reinforce idea #####
#
#         gradients_enc = tape.gradient(loss, self.model['enc'].trainable_weights)
#
#         gradients_enc, grad_norm_enc = tf.clip_by_global_norm(gradients_enc, self.config.clip_grad_norm_enc)
#
#         with self.config.train_writer.as_default():  # update trainer metrics
#             tf.summary.scalar("grad_norm_enc", grad_norm_enc, self.global_step)
#             tf.summary.scalar("mdp_loss", loss, self.global_step)
#             self.global_step.assign_add(1)
#
#         if self.feedback:
#             x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             y_ = tf.split(y, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             self.fb_input = tf.concat([x_, y_], axis=-1)
#         else:
#             x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             self.fb_input = x_
#
#         return gradients_enc, t_y, t_xy, p
#
#     # @tf.function
#     def calc_enc_loss_input(self, t_l, x, p):
#         p_et = tf.stack(self.config.contrastive_duplicates * [p], axis=2)
#         p_bar_et = tf.ones_like(p_et) - p_et
#         p_t = tf.expand_dims(p, axis=2)
#         p_bar_t = tf.ones_like(p_t) - p_t  # obtain p_bar values
#         [t, et] = t_l
#         et_mean = tf.reduce_mean(et)
#
#         t_0, t_1 = t * tf.math.log(p_bar_t), t * tf.math.log(p_t)  # create tensor to draw values from
#         et_0, et_1 = et * tf.math.log(p_bar_et), et * tf.math.log(p_et)
#
#         x_t = tf.expand_dims(x, axis=2)
#         x_et = tf.stack(self.config.contrastive_duplicates * [x], axis=2)
#
#         t_enc = tf.where(tf.equal(x_t, 1), t_1, t_0)  # choose according to binary criteria
#         et_enc = tf.where(tf.equal(x_et, 1), et_1, et_0)
#
#         return [t_enc, et_enc / et_mean]
#
#     def enc_lp(self, x, p):
#         # p_et = tf.stack(self.config.contrastive_duplicates * [p], axis=2)
#         # p_bar_et = tf.ones_like(p_et) - p_et
#         # p_t = tf.expand_dims(p, axis=2)
#         # p_bar_t = tf.ones_like(p_t) - p_t  # obtain p_bar values
#         # [t, et] = t_l
#         # et_mean = tf.reduce_mean(et)
#         #
#         # t_0, t_1 = t * tf.math.log(p_bar_t), t * tf.math.log(p_t)  # create tensor to draw values from
#         # et_0, et_1 = et * tf.math.log(p_bar_et), et * tf.math.log(p_et)
#         #
#         # x_t = tf.expand_dims(x, axis=2)
#         # x_et = tf.stack(self.config.contrastive_duplicates * [x], axis=2)
#         #
#         # t_enc = tf.where(tf.equal(x_t, 1), t_1, t_0)  # choose according to binary criteria
#         # et_enc = tf.where(tf.equal(x_et, 1), et_1, et_0)
#         p_out = tf.math.log(tf.where(tf.equal(x,1), p, 1-p))
#         return p_out
#
#     @tf.function
#     def apply_enc_grads(self, gradients_enc):
#         self.optimizer['enc'].apply_gradients(zip(gradients_enc, self.model['enc'].trainable_weights))
#
#     def sync_eval_model(self):
#         # sync DV:
#         w_y = self.model['dv']['y'].get_weights()
#         w_xy = self.model['dv']['xy'].get_weights()
#         self.model['dv_eval']['y'].set_weights(w_y)
#         self.model['dv_eval']['xy'].set_weights(w_xy)
#         # sync enc:
#         w_enc = self.model['enc'].get_weights()  # similarly sync encoder model
#         self.model['enc_eval'].set_weights(w_enc)
#
#     def evaluate(self, epoch, iterator="eval"):
#         self.sync_eval_model()
#         self.metrics["eval"].reset_states()
#         self.reset_model_states()
#         self.saver.reset_state()
#         i=0
#         p_t = []
#         t_y_t = []
#         states = []
#
#         for input_batch in self.data_iterators[iterator]():
#             input_batch = self.fb_input
#             output = self.eval_step(input_batch)
#             p_t.append(output[2])
#             t_y_t.append(output[0][0])
#             states.append(output[3])
#             self.metrics["eval"].update_state(output[:-1])
#
#         self.metrics["eval"].log_metrics(epoch, model_name="Evaluation")
#         p_t = tf.concat(p_t, axis=1)
#         t_y_t = tf.concat(t_y_t, axis=1)
#         states = tf.concat(states, axis=1)
#         with self.config.test_writer.as_default():
#             tf.summary.histogram(name="p_hist", data=p_t, step=self.global_step, buckets=20)
#             tf.summary.histogram(name="t_y_hist", data=t_y_t, step=self.global_step, buckets=20)
#             tf.summary.histogram(name="states_y_hist", data=states, step=self.global_step, buckets=20)
#
#         self.saver.save(None if epoch > 0 else 'begin.mat')
#         if self.config.trainer_name == "di_with_enc_states":
#             self.saver.save_models(models=self.model, path=os.path.join(self.config.tensor_board_dir, "models_weights"))
#
#     # @tf.function
#     def eval_step(self, input_batch):
#         [x, y, p, s] = self.model['enc'](input_batch, training=False)
#         xy = tf.concat([x, y], axis=-1)
#         t_y = self.model['dv_eval']['y'](y, training=False)
#         t_xy = self.model['dv_eval']['xy'](xy, training=False)
#         # addition with states (now output is [t,et,s]:
#         states_y = t_y[2]
#         t_y = [t_y[0], t_y[1]]
#         t_xy = [t_xy[0], t_xy[1]]
#
#         self.saver.update_state(t_y, t_xy, x, y, p, states_y)
#
#         if self.feedback:
#             x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             y_ = tf.split(y, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             self.fb_input = tf.concat([x_, y_], axis=-1)
#         else:
#             x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             self.fb_input = x_
#
#         return t_y, t_xy, p, states_y
#
#     def train(self):
#         for epoch in range(self.config.num_epochs):
#             if epoch % self.config.eval_freq == 0:
#                 self.evaluate(epoch)  # perform evaluation step
#                 continue
#             self.train_epoch(epoch)  # perform a training epoch
#
#         self.evaluate(self.config.num_epochs, iterator="long_eval")  # perform a long evaluation
#
#     def reset_model_states(self):
#         def reset_recursively(models):
#             for model in models.values():
#                 if isinstance(model, dict):
#                     reset_recursively(model)
#                 else:
#                     model.reset_states()
#
#         reset_recursively(self.model)
#         if (self.config.channel_name == "ising") or (self.config.channel_name == "trapdoor"):
#             self.model['enc'].layers[3].reset_states()
#             self.model['enc_eval'].layers[3].reset_states()
#
#     @tf.function
#     def apply_grads(self, gradients_dv_y, gradients_dv_xy):
#         self.optimizer['dv_y'].apply_gradients(zip(gradients_dv_y, self.model['dv']['y'].trainable_weights))
#         self.optimizer['dv_xy'].apply_gradients(zip(gradients_dv_xy, self.model['dv']['xy'].trainable_weights))
#
#
# class PDINE_trainer(object):
#     def __init__(self, model, data, config):
#         """
#         Constructor of trainer class, inherits from DINumericalTrainer
#         :param model: a dictionary consisting of all models relevant to this trainer
#         :param data: data loader instance, not used in this class.
#         :param config: configuration parameters obtains for the relevant .json file
#         """
#         # general assignments:
#         self.model = model  # the DINE model
#         self.data_iterators = data  # the data generator
#         self.config = config  # the configuration
#
#         # models losses:
#         self.loss_fn = Model_loss()  # the DINE model loss function
#         self.enc_loss_fn = Enc_Loss_Reinforce(config=config)  # encoder loss function
#         self.q_loss_function = CategoricalCrossentropy()  # q-graph estimator loss function
#
#         # learning rate:
#         self.learning_rate = config.lr  # the model's learning rate
#         if config.decay == 1:
#             self.learning_rate_dv = exp_dec_lr(config, data, config.lr)
#             self.learning_rate_enc = exp_dec_lr(config, data, config.lr/5)
#             self.learning_rate_q = exp_dec_lr(config, data, config.lr/4)
#         else:
#             self.learning_rate_dv = config.lr
#             self.learning_rate_enc = config.lr/4
#             self.learning_rate_q = config.lr / 4
#
#         self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
#
#         self.bptt = config.bptt
#         if config.optimizer == "adam":
#             self.optimizer = {"dv_y": Adam(amsgrad=True, learning_rate=self.learning_rate_dv),
#                               "dv_xy": Adam(amsgrad=True, learning_rate=self.learning_rate_dv),
#                               # "enc": Adam(amsgrad=True, learning_rate=self.learning_rate_enc),  # the model's optimizers,
#                               "enc": Adam(amsgrad=False, learning_rate=self.learning_rate_enc),  # the model's optimizers,
#                               "q_graph": Adam(amsgrad=True, learning_rate=self.learning_rate_q)}
#         else:
#             self.optimizer = {"dv_y": SGD(learning_rate=config.lr_SGD),
#                               "dv_xy": SGD(learning_rate=config.lr_SGD),
#                               "enc": SGD(learning_rate=config.lr_SGD/4),
#                               "q": SGD(learning_rate=config.lr_SGD/4)
#                               }  # the model's optimizers
#
#         self.saver = DVEncoderVisualizer_states(config)
#         self.fsc_flag = self.check_fsc(config)
#
#
#         # metrics:
#         self.global_step_enc = tf.Variable(0, trainable=False, dtype=tf.int64)
#         if self.fsc_flag:
#             self.metrics = {"train": PDINE_metrics(config.train_writer, name='dv_train'),
#                             "eval": PDINE_metrics(config.test_writer, name='dv_eval')}  # the trainer's metrics
#         else:
#             self.metrics = {"train": ModelWithEncMetrics(config.train_writer, name='dv_train'),
#                             "eval": ModelWithEncMetrics(config.test_writer, name='dv_eval')}  # the trainer's metrics
#
#         # additional actions:
#         self.feedback = (config.feedback == 1)
#         self.T = config.T
#         self.train_q = False
#         if self.feedback:
#             self.fb_input = tf.zeros(shape=[config.batch_size, 1, config.x_dim + config.y_dim])
#         else:
#             self.fb_input = tf.zeros(shape=[config.batch_size, 1, config.x_dim])
#
#         self.reset_channel = config.reset_channel==1
#
#     def train(self):
#         for epoch in range(self.config.num_epochs):
#             if epoch % self.config.eval_freq == 0:
#                 self.evaluate(epoch)  # perform evaluation step
#                 continue
#             self.train_epoch(epoch)  # perform a training epoch
#
#             if epoch > 2 and self.fsc_flag:
#                 # self.train_q = True
#                 self.train_q = False
#
#         self.evaluate(self.config.num_epochs, iterator="long_eval")  # perform a long evaluation
#
#     def evaluate(self, epoch, iterator="eval"):
#         self.sync_eval_model()
#         self.metrics["eval"].reset_states()
#         self.reset_model_states()
#         self.saver.reset_state()
#         p_t = []
#         t_y_t = []
#         states = []
#         q_t = []
#
#         for input_batch in self.data_iterators[iterator]():
#             input_batch = self.fb_input
#             output = self.eval_step(input_batch)
#             p_t.append(output[2])
#             t_y_t.append(output[0][0])
#             states.append(output[3][0])
#             q_t.append(output[3][1])
#             if self.fsc_flag:
#                 self.metrics["eval"].update_state(output[0:3]+output[4:])
#             else:
#                 self.metrics["eval"].update_state(output[:-1])
#
#         self.metrics["eval"].log_metrics(epoch, model_name="Evaluation")
#         p_t = tf.concat(p_t, axis=1)
#         t_y_t = tf.concat(t_y_t, axis=1)
#         states = tf.concat(states, axis=1)
#         with self.config.test_writer.as_default():
#             tf.summary.histogram(name="p_hist", data=p_t, step=self.global_step, buckets=20)
#             tf.summary.histogram(name="t_y_hist", data=t_y_t, step=self.global_step, buckets=20)
#             # tf.summary.histogram(name="states_y_hist", data=states, step=self.global_step, buckets=20)
#
#         name=None if epoch > 0 else 'begin.mat'
#         self.saver.save(name=name,models=self.model)
#
#     def eval_step(self, input_batch):
#         if self.fsc_flag:
#             [x, y, p, s] = self.model['enc'](input_batch, training=False)  # obtain samples through model
#             q_est = self.model['q_graph'](y)
#             s_onehot = to_categorical(s, num_classes=2)
#             self.saver.histogram(q_est)
#         else:
#             [x, y, p] = self.model['enc'](input_batch, training=False)  # obtain samples through model
#         xy = tf.concat([x, y], axis=-1)
#         t_y = self.model['dv_eval']['y'](y, training=False)
#         t_xy = self.model['dv_eval']['xy'](xy, training=False)
#         # addition with states (now output is [t,et,s]:
#         states_y = t_y[2]
#         t_y = [t_y[0], t_y[1]]
#         t_xy = [t_xy[0], t_xy[1]]
#         if self.fsc_flag:
#             states_y = [states_y, q_est]
#
#
#         self.saver.update_state(t_y, t_xy, x, y, p, states_y)
#
#         if self.feedback:
#             x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             y_ = tf.split(y, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             self.fb_input = tf.concat([x_, y_], axis=-1)
#         else:
#             x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             self.fb_input = x_
#
#         if self.fsc_flag:
#             return [t_y, t_xy, p, states_y, q_est, s_onehot]
#         else:
#             return [t_y, t_xy, p, states_y]
#
#     def train_epoch(self, epoch):
#         self.metrics["train"].reset_states()
#         self.reset_model_states()  # reset model and metrics states
#         i = 0
#         if np.random.rand() > 0.3 or epoch < 20:  # train DINE
#             model_name = "DV epoch"
#             for sample in self.data_iterators["train"]():
#                 # if sample is None:
#                 #     self.reset_model_states()  # in case we deal with a dataset
#                 #     continue
#                 # if i > 0:
#                 sample = self.fb_input
#                 output = self.train_dine_step(sample)  # calculate model outputs and perform a training step
#                 self.metrics["train"].update_state(output)  # update trainer metrics
#         else:  # train Enc.
#             model_name = "Encoder epoch"
#             for sample in self.data_iterators["train"]():
#                 output = self.train_enc_step(self.fb_input)  # calculate model outputs and perform a training step
#                 self.metrics["train"].update_state(output)  # update trainer metrics
#
#         self.metrics["train"].log_metrics(epoch, model_name=model_name)  # log updated metrics
#
#     def train_dine_step(self, sample):
#         if self.fsc_flag:
#             if self.train_q:
#                 [[gradients_dv_y, gradients_dv_xy,gradients_q], data_metrics] = self.compute_dine_grads(sample)
#                 self.apply_q_grads(gradients_q)
#                 self.apply_dv_grads(gradients_dv_y, gradients_dv_xy)  # apply gradients
#                 return data_metrics
#             else:
#                 [[gradients_dv_y, gradients_dv_xy], data_metrics] = self.compute_dine_grads(sample)
#                 self.apply_dv_grads(gradients_dv_y, gradients_dv_xy)  # apply gradients
#                 return data_metrics
#         else:
#             [[gradients_dv_y, gradients_dv_xy], data_metrics] = self.compute_dine_grads(sample)  # calculate gradients
#             self.apply_dv_grads(gradients_dv_y, gradients_dv_xy)  # apply gradients
#             return data_metrics
#
#     def train_enc_step(self, sample):
#         [gradients_enc, data_metrics] = self.compute_enc_grads(sample)  # calculate gradients
#         self.apply_enc_grads(gradients_enc)  # apply gradients
#         return data_metrics
#
#     # @tf.function
#     def compute_dine_grads(self, sample):  # gradients computation method
#         with tf.GradientTape(persistent=True) as tape:
#             if self.fsc_flag:
#                 [x, y, p, s] = self.model['enc'](sample, training=False)  # obtain samples through model
#                 q_est = self.model['q_graph'](y)
#                 s_one_hot = to_categorical(s, num_classes=2)
#             else:
#                 [x, y, p] = self.model['enc'](sample, training=False)  # obtain samples through model
#             xy = tf.concat([x, y], axis=-1)  # t_xy model input
#             t_y = self.model['dv']['y'](y, training=True)
#             t_xy = self.model['dv']['xy'](xy, training=True)  # calculate models outputs
#             loss = self.loss_fn(t_y, t_xy)  # calculate loss for each model
#             loss_y, loss_xy = tf.split(loss, num_or_size_splits=2, axis=0)
#             if self.train_q:
#                 loss_q = self.q_loss_function(s_one_hot, q_est)
#
#         gradients_dv_y = tape.gradient(loss_y, self.model['dv']['y'].trainable_weights)
#         gradients_dv_xy = tape.gradient(loss_xy, self.model['dv']['xy'].trainable_weights)  # calculate gradients
#
#         gradients_dv_y, grad_norm_dv_y = tf.clip_by_global_norm(gradients_dv_y, self.config.clip_grad_norm)
#         gradients_dv_xy, grad_norm_dv_xy = tf.clip_by_global_norm(gradients_dv_xy, self.config.clip_grad_norm)  # normalize gradients
#
#         if self.train_q:
#             gradients_q = tape.gradient(loss_q, self.model['q_graph'].trainable_weights)
#             gradients_q, grad_norm_q = tf.clip_by_global_norm(gradients_q, self.config.clip_grad_norm_q)
#
#         with self.config.train_writer.as_default():  # update trainer metrics
#             tf.summary.scalar("grad_norm_dv_y", grad_norm_dv_y, self.global_step)
#             tf.summary.scalar("grad_norm_dv_xy", grad_norm_dv_xy, self.global_step)
#             tf.summary.scalar("x_mean", tf.math.reduce_mean(x), self.global_step)
#             tf.summary.scalar("y_mean", tf.math.reduce_mean(y), self.global_step)
#             if self.train_q:
#                 tf.summary.scalar("grad_norm_q", grad_norm_q, self.global_step)
#
#             self.global_step.assign_add(1)
#
#         if self.feedback:
#             x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             y_ = tf.split(y, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             self.fb_input = tf.concat([x_, y_], axis=-1)
#         else:
#             x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             self.fb_input = x_
#
#
#         if self.fsc_flag:
#             if self.train_q:
#                 return [[gradients_dv_y, gradients_dv_xy, gradients_q], [t_y, t_xy, p, q_est, s_one_hot]]
#             else:
#                 return [[gradients_dv_y, gradients_dv_xy], [t_y, t_xy, p, q_est, s_one_hot]]
#         else:
#             return [[gradients_dv_y, gradients_dv_xy], [t_y, t_xy, p]]
#
#     def compute_enc_grads(self, sample):  # gradients computation method
#         with tf.GradientTape(persistent=True) as tape:
#             if self.fsc_flag:
#                 [x, y, p, s] = self.model['enc'](sample, training=False)  # obtain samples through model
#                 q_est = self.model['q_graph'](y)
#                 s_one_hot = to_categorical(s, num_classes=2)
#             else:
#                 [x, y, p] = self.model['enc'](sample, training=False)  # obtain samples through model
#             xy = tf.concat([x, y], axis=-1)  # t_xy model input
#             t_y = self.model['dv']['y'](y, training=False)
#             t_xy = self.model['dv']['xy'](xy, training=False)  # calculate models outputs
#             loss = self.enc_loss_fn([x, p], [t_xy[0], t_y[0]])
#
#         gradients_enc = tape.gradient(loss, self.model['enc'].trainable_weights)
#
#         gradients_enc, grad_norm_enc = tf.clip_by_global_norm(gradients_enc, self.config.clip_grad_norm_enc)
#
#         with self.config.train_writer.as_default():  # update trainer metrics
#             tf.summary.scalar("grad_norm_enc", grad_norm_enc, self.global_step)
#             tf.summary.scalar("mdp_loss", loss, self.global_step)
#             self.global_step.assign_add(1)
#
#         if self.feedback:
#             x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             y_ = tf.split(y, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             self.fb_input = tf.concat([x_, y_], axis=-1)
#         else:
#             x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             self.fb_input = x_
#
#         if self.fsc_flag:
#             return [gradients_enc, [t_y, t_xy, p, q_est, s_one_hot]]
#         else:
#             return [gradients_enc, [t_y, t_xy, p]]
#
#     def apply_dv_grads(self, gradients_dv_y, gradients_dv_xy):
#         self.optimizer['dv_y'].apply_gradients(zip(gradients_dv_y, self.model['dv']['y'].trainable_weights))
#         self.optimizer['dv_xy'].apply_gradients(zip(gradients_dv_xy, self.model['dv']['xy'].trainable_weights))
#
#     def apply_enc_grads(self, gradients_enc):
#         self.optimizer['enc'].apply_gradients(zip(gradients_enc, self.model['enc'].trainable_weights))
#
#     def apply_q_grads(self, gradients_q):
#         self.optimizer['q_graph'].apply_gradients(zip(gradients_q, self.model['q_graph'].trainable_weights))
#
#     def sync_eval_model(self):
#         # sync DV model:
#         w_y = self.model['dv']['y'].get_weights()
#         w_xy = self.model['dv']['xy'].get_weights()
#         self.model['dv_eval']['y'].set_weights(w_y)
#         self.model['dv_eval']['xy'].set_weights(w_xy)
#
#     def reset_model_states(self):
#         def reset_recursively(models):
#             for model in models.values():
#                 if isinstance(model, dict):
#                     reset_recursively(model)
#                 else:
#                     model.reset_states()
#
#         reset_recursively(self.model)
#
#         if self.reset_channel:
#             if (self.config.channel_name == "ising") or (self.config.channel_name == "trapdoor"):
#                 self.model['enc'].layers[3].reset_states()
#                 self.model['enc_eval'].layers[3].reset_states()
#
#     def check_fsc(self,config):
#         return config.channel_name in ['ising','trapdoor','post','GE']


# class PDINE_trainer_no_q(object):
#     def __init__(self, model, data, config):
#         """
#         Constructor of trainer class, inherits from DINumericalTrainer
#         :param model: a dictionary consisting of all models relevant to this trainer
#         :param data: data loader instance, not used in this class.
#         :param config: configuration parameters obtains for the relevant .json file
#         """
#         # general assignments:
#         self.model = model  # the DINE model
#         self.data_iterators = data  # the data generator
#         self.config = config  # the configuration
#
#         # models losses:
#         self.loss_fn = Model_loss()  # the DINE model loss function
#         self.enc_loss_fn = Enc_Loss_Reinforce(config=config)  # encoder loss function
#
#         # learning rate:
#         self.learning_rate = config.lr  # the model's learning rate
#         if config.decay == 1:
#             self.learning_rate_dv = exp_dec_lr(config, data, config.lr)
#             self.learning_rate_enc = exp_dec_lr(config, data, config.lr/5)
#         else:
#             self.learning_rate_dv = config.lr
#             self.learning_rate_enc = config.lr/4
#
#         self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
#
#         self.bptt = config.bptt
#         if config.optimizer == "adam":
#             self.optimizer = {"dv_y": Adam(amsgrad=True, learning_rate=self.learning_rate_dv),
#                               "dv_xy": Adam(amsgrad=True, learning_rate=self.learning_rate_dv),
#                               "enc": Adam(amsgrad=False, learning_rate=self.learning_rate_enc)}
#         else:
#             self.optimizer = {"dv_y": SGD(learning_rate=config.lr_SGD),
#                               "dv_xy": SGD(learning_rate=config.lr_SGD),
#                               "enc": SGD(learning_rate=config.lr_SGD/4)}  # the model's optimizers
#
#         self.saver = DVEncoderVisualizer_states(config)
#         self.fsc_flag = self.check_fsc(config)
#
#         # metrics:
#         self.global_step_enc = tf.Variable(0, trainable=False, dtype=tf.int64)
#         if self.fsc_flag:
#             self.metrics = {"train": PDINE_metrics(config.train_writer, name='dv_train'),
#                             "eval": PDINE_metrics(config.test_writer, name='dv_eval')}  # the trainer's metrics
#         else:
#             self.metrics = {"train": ModelWithEncMetrics(config.train_writer, name='dv_train'),
#                             "eval": ModelWithEncMetrics(config.test_writer, name='dv_eval')}  # the trainer's metrics
#
#         # additional actions:
#         self.feedback = (config.feedback == 1)
#         self.T = config.T
#         if self.feedback:
#             self.fb_input = tf.zeros(shape=[config.batch_size, 1, config.x_dim + config.y_dim])
#         else:
#             self.fb_input = tf.zeros(shape=[config.batch_size, 1, config.x_dim])
#
#         self.reset_channel = config.reset_channel==1
#
#     def train(self):
#         for epoch in range(self.config.num_epochs):
#             if epoch % self.config.eval_freq == 0:
#                 self.evaluate(epoch)  # perform evaluation step
#                 continue
#             self.train_epoch(epoch)  # perform a training epoch
#
#         self.evaluate(self.config.num_epochs, iterator="long_eval")  # perform a long evaluation
#
#     def evaluate(self, epoch, iterator="eval"):
#         self.sync_eval_model()
#         self.metrics["eval"].reset_states()
#         self.reset_model_states()
#         self.saver.reset_state()
#         p_t = []
#
#
#         for input_batch in self.data_iterators[iterator]():
#             input_batch = self.fb_input
#             output = self.eval_step(input_batch)
#             p_t.append(output[2])
#             if self.fsc_flag:
#                 self.metrics["eval"].update_state(output)
#             else:
#                 self.metrics["eval"].update_state(output)
#
#         self.metrics["eval"].log_metrics(epoch, model_name="Evaluation")
#         p_t = tf.concat(p_t, axis=1)
#         with self.config.test_writer.as_default():
#             tf.summary.histogram(name="p_hist", data=p_t, step=self.global_step, buckets=20)
#
#         name=None if epoch > 0 else 'begin.mat'
#         self.saver.save(name=name,models=self.model)
#
#     def eval_step(self, input_batch):
#         if self.fsc_flag:
#             [x, y, p, s] = self.model['enc'](input_batch, training=False)  # obtain samples through model
#         else:
#             [x, y, p] = self.model['enc'](input_batch, training=False)  # obtain samples through model
#         xy = tf.concat([x, y], axis=-1)
#         t_y = self.model['dv_eval']['y'](y, training=False)
#         t_xy = self.model['dv_eval']['xy'](xy, training=False)
#         # addition with states (now output is [t,et,s]:
#         t_y = [t_y[0], t_y[1]]
#         t_xy = [t_xy[0], t_xy[1]]
#
#
#         self.saver.update_state(t_y, t_xy, x, y, p)
#
#         if self.feedback:
#             x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             y_ = tf.split(y, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             self.fb_input = tf.concat([x_, y_], axis=-1)
#         else:
#             x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             self.fb_input = x_
#
#         if self.fsc_flag:
#             return [t_y, t_xy, p]
#         else:
#             return [t_y, t_xy, p]
#
#     def train_epoch(self, epoch):
#         self.metrics["train"].reset_states()
#         self.reset_model_states()  # reset model and metrics states
#         i = 0
#         if np.random.rand() > 0.3 or epoch < 20:  # train DINE
#             model_name = "DV epoch"
#             for sample in self.data_iterators["train"]():
#                 sample = self.fb_input
#                 output = self.train_dine_step(sample)  # calculate model outputs and perform a training step
#                 self.metrics["train"].update_state(output)  # update trainer metrics
#         else:  # train Enc.
#             model_name = "Encoder epoch"
#             for sample in self.data_iterators["train"]():
#                 output = self.train_enc_step(self.fb_input)  # calculate model outputs and perform a training step
#                 self.metrics["train"].update_state(output)  # update trainer metrics
#
#         self.metrics["train"].log_metrics(epoch, model_name=model_name)  # log updated metrics
#
#     def train_dine_step(self, sample):
#         if self.fsc_flag:
#             [[gradients_dv_y, gradients_dv_xy], data_metrics] = self.compute_dine_grads(sample)
#             self.apply_dv_grads(gradients_dv_y, gradients_dv_xy)  # apply gradients
#             return data_metrics
#         else:
#             [[gradients_dv_y, gradients_dv_xy], data_metrics] = self.compute_dine_grads(sample)  # calculate gradients
#             self.apply_dv_grads(gradients_dv_y, gradients_dv_xy)  # apply gradients
#             return data_metrics
#
#     def train_enc_step(self, sample):
#         [gradients_enc, data_metrics] = self.compute_enc_grads(sample)  # calculate gradients
#         self.apply_enc_grads(gradients_enc)  # apply gradients
#         return data_metrics
#
#     # @tf.function
#     def compute_dine_grads(self, sample):  # gradients computation method
#         with tf.GradientTape(persistent=True) as tape:
#             if self.fsc_flag:
#                 [x, y, p, s] = self.model['enc'](sample, training=False)  # obtain samples through model
#             else:
#                 [x, y, p] = self.model['enc'](sample, training=False)  # obtain samples through model
#             xy = tf.concat([x, y], axis=-1)  # t_xy model input
#             t_y = self.model['dv']['y'](y, training=True)
#             t_xy = self.model['dv']['xy'](xy, training=True)  # calculate models outputs
#             loss = self.loss_fn(t_y, t_xy)  # calculate loss for each model
#             loss_y, loss_xy = tf.split(loss, num_or_size_splits=2, axis=0)
#
#         gradients_dv_y = tape.gradient(loss_y, self.model['dv']['y'].trainable_weights)
#         gradients_dv_xy = tape.gradient(loss_xy, self.model['dv']['xy'].trainable_weights)  # calculate gradients
#
#         gradients_dv_y, grad_norm_dv_y = tf.clip_by_global_norm(gradients_dv_y, self.config.clip_grad_norm)
#         gradients_dv_xy, grad_norm_dv_xy = tf.clip_by_global_norm(gradients_dv_xy, self.config.clip_grad_norm)  # normalize gradients
#
#         with self.config.train_writer.as_default():  # update trainer metrics
#             tf.summary.scalar("grad_norm_dv_y", grad_norm_dv_y, self.global_step)
#             tf.summary.scalar("grad_norm_dv_xy", grad_norm_dv_xy, self.global_step)
#             tf.summary.scalar("x_mean", tf.math.reduce_mean(x), self.global_step)
#             tf.summary.scalar("y_mean", tf.math.reduce_mean(y), self.global_step)
#             self.global_step.assign_add(1)
#
#         if self.feedback:
#             x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             y_ = tf.split(y, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             self.fb_input = tf.concat([x_, y_], axis=-1)
#         else:
#             x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             self.fb_input = x_
#
#
#         if self.fsc_flag:
#             return [[gradients_dv_y, gradients_dv_xy], [t_y, t_xy, p, q_est, s_one_hot]]
#         else:
#             return [[gradients_dv_y, gradients_dv_xy], [t_y, t_xy, p]]
#
#     def compute_enc_grads(self, sample):  # gradients computation method
#         with tf.GradientTape(persistent=True) as tape:
#             if self.fsc_flag:
#                 [x, y, p, s] = self.model['enc'](sample, training=False)  # obtain samples through model
#             else:
#                 [x, y, p] = self.model['enc'](sample, training=False)  # obtain samples through model
#             xy = tf.concat([x, y], axis=-1)  # t_xy model input
#             t_y = self.model['dv']['y'](y, training=False)
#             t_xy = self.model['dv']['xy'](xy, training=False)  # calculate models outputs
#             loss = self.enc_loss_fn([x, p], [t_xy[0], t_y[0]])
#
#         gradients_enc = tape.gradient(loss, self.model['enc'].trainable_weights)
#
#         gradients_enc, grad_norm_enc = tf.clip_by_global_norm(gradients_enc, self.config.clip_grad_norm_enc)
#
#         with self.config.train_writer.as_default():  # update trainer metrics
#             tf.summary.scalar("grad_norm_enc", grad_norm_enc, self.global_step)
#             tf.summary.scalar("mdp_loss", loss, self.global_step)
#             self.global_step.assign_add(1)
#
#         if self.feedback:
#             x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             y_ = tf.split(y, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             self.fb_input = tf.concat([x_, y_], axis=-1)
#         else:
#             x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             self.fb_input = x_
#
#         if self.fsc_flag:
#             return [gradients_enc, [t_y, t_xy, p, q_est, s_one_hot]]
#         else:
#             return [gradients_enc, [t_y, t_xy, p]]
#
#     def apply_dv_grads(self, gradients_dv_y, gradients_dv_xy):
#         self.optimizer['dv_y'].apply_gradients(zip(gradients_dv_y, self.model['dv']['y'].trainable_weights))
#         self.optimizer['dv_xy'].apply_gradients(zip(gradients_dv_xy, self.model['dv']['xy'].trainable_weights))
#
#     def apply_enc_grads(self, gradients_enc):
#         self.optimizer['enc'].apply_gradients(zip(gradients_enc, self.model['enc'].trainable_weights))
#
#     def sync_eval_model(self):
#         # sync DV model:
#         w_y = self.model['dv']['y'].get_weights()
#         w_xy = self.model['dv']['xy'].get_weights()
#         self.model['dv_eval']['y'].set_weights(w_y)
#         self.model['dv_eval']['xy'].set_weights(w_xy)
#
#     def reset_model_states(self):
#         def reset_recursively(models):
#             for model in models.values():
#                 if isinstance(model, dict):
#                     reset_recursively(model)
#                 else:
#                     model.reset_states()
#
#         reset_recursively(self.model)
#
#         if self.reset_channel:
#             if (self.config.channel_name == "ising") or (self.config.channel_name == "trapdoor"):
#                 self.model['enc'].layers[3].reset_states()
#                 self.model['enc_eval'].layers[3].reset_states()
#
#     def check_fsc(self,config):
#         return config.channel_name in ['ising','trapdoor','post','GE']

#
# class Q_est_trainer(object):
#     def __init__(self, model, data, config):
#         self.config = config
#         self.model = model
#         self.channel_name = config.channel_name
#         self.load_enc_weights()
#         self.data_iterator = data
#         self.loss_function = CategoricalCrossentropy()  # q-graph estimator loss function
#
#         # learning rate:
#         self.learning_rate = config.lr  # the model's learning rate
#         if config.decay == 1:
#             self.learning_rate = exp_dec_lr(config, data, config.lr / 4)
#         else:
#             self.learning_rate = config.lr / 4
#
#         self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
#         self.bptt = config.bptt
#         if config.optimizer == "adam":
#             self.optimizer = {"q_graph": Adam(amsgrad=True, learning_rate=self.learning_rate)}
#         else:
#             self.optimizer = {"q_graph": SGD(learning_rate=config.lr_SGD / 4)}  # the model's optimizers
#
#         self.saver = Q_est_saver(config)
#
#         # metrics:
#         self.global_step_enc = tf.Variable(0, trainable=False, dtype=tf.int64)
#         self.metrics = {"train": Q_est_metrics(config.train_writer, name='dv_train'),
#                             "eval": Q_est_metrics(config.test_writer, name='dv_eval')}
#
#         self.fb_input = tf.zeros(shape=[config.batch_size, 1, config.x_dim + config.y_dim])
#
#         # external data generator:
#         self.external_gen = False
#         if self.external_gen:
#             self.data_gen = IsingChannel_state(bptt=config.bptt, input_shape=[config.batch_size,1])
#             self.data_gen._call_()
#
#     def evaluate(self, epoch, iterator="eval"):
#         self.sync_eval_model()
#         self.metrics["eval"].reset_states()
#         self.reset_model_states()
#         self.saver.reset_state()
#         s_t = []
#         q_t = []
#
#         for input_batch in self.data_iterator[iterator]():
#             input_batch = self.fb_input
#             output = self.eval_step(input_batch)  # output = [q_est,s_onehot]
#             q_t.append(output[0])
#             s_t.append(output[1])
#             self.metrics["eval"].update_state(output)
#
#         self.metrics["eval"].log_metrics(epoch)
#         q_t = tf.concat(q_t, axis=1)
#         s_t = tf.concat(s_t, axis=1)
#         with self.config.test_writer.as_default():
#             tf.summary.histogram(name="Q_0 hist", data=q_t[:-1,0], step=self.global_step, buckets=50)
#             tf.summary.histogram(name="Q_1 hist", data=q_t[:-1,1], step=self.global_step, buckets=50)
#
#         name=None if epoch > 0 else 'begin.mat'
#         self.saver.save(name=name)
#         self.saver.save_models(models=self.model, path=self.config.tensor_board_dir)
#
#     def eval_step(self, input_batch):
#         if self.external_gen:
#             [x, y] = self.data_gen._gen_()
#             s = x
#         else:
#             [x, y, p, s] = self.model['enc'](input_batch, training=False)  # obtain samples through model
#         # [x, y, p, s] = self.model['enc'](input_batch, training=False)  # obtain samples through model
#         q_est = self.model['q_graph_test'](y, training=True)
#         s_onehot = to_categorical(s, num_classes=2)
#         # self.saver.histogram(q_est)
#
#         self.saver.update_state([q_est, s_onehot, y])
#
#         x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
#         y_ = tf.split(y, axis=1, num_or_size_splits=self.config.bptt)[-1]
#         self.fb_input = tf.concat([x_, y_], axis=-1)
#
#         return [q_est, s_onehot]
#
#     def load_enc_weights(self):
#         """
#         model to set weights is encoder
#         """
#         # if self.channel_name == 'ising':
#         #     p_val = self.config.p_ising
#         # elif self.channel_name == 'trapdoor':
#         #     p_val = self.config.p_trapdoor
#         # elif self.channel_name == 'post':
#         #     p_val = self.config.p_post
#         # else:
#         #     p_val = 0.5
#         # dir_name = "enc_" + self.config.channel_name + str(p_val)
#         # dir_name = os.path.join("pretrained_models_weights","enc_ising_01/enc.index")
#         # weights_path = os.path.join(os.getcwd(),dir_name)
#         # self.model['enc'].load_weights(weights_path)
#         # for loading model:
#         # loading model:
#         # self.model['enc'] = None
#         # filepath = os.path.join("./pretrained_models",self.config.channel_name+"_fb","p_05_4037_47k")
#         # self.model['enc'] = tf.keras.models.load_model(filepath=filepath)
#
#     #     loading weights:
#         filepath = os.path.join("./pretrained_models", self.config.channel_name + "_fb", "p_02_emp_","weights")
#         self.model['enc'].load_weights(filepath)
#
#     def train(self):
#         for epoch in range(self.config.num_epochs):
#             if epoch % self.config.eval_freq == 0:
#                 self.evaluate(epoch)  # perform evaluation step
#                 continue
#             self.train_epoch(epoch)  # perform a training epoch
#
#         self.evaluate(self.config.num_epochs, iterator="long_eval")  # perform a long evaluation
#
#     def train_epoch(self, epoch):
#         self.metrics["train"].reset_states()
#         self.reset_model_states()  # reset model and metrics states
#         for sample in self.data_iterator["train"]():
#             sample = self.fb_input
#             output = self.train_step(sample)  # calculate model outputs and perform a training step
#             self.metrics["train"].update_state(output)  # update trainer metrics
#
#         self.metrics["train"].log_metrics(epoch)  # log updated metrics
#
#     def train_step(self, sample):
#         gradients, data_metrics = self.compute_grads(sample)
#         self.apply_grads(gradients)
#         return data_metrics
#
#     def compute_grads(self, sample):  # gradients computation method
#         with tf.GradientTape(persistent=True) as tape:
#             if self.external_gen:
#                 [x, y] = self.data_gen._gen_()
#                 s = x
#             else:
#                 [x, y, p, s] = self.model['enc'](sample, training=False)  # obtain samples through model
#             q_est = self.model['q_graph'](y, training=True)
#             s_onehot = to_categorical(s, num_classes=2)
#             loss = self.loss_function(s_onehot, q_est)
#
#         gradients = tape.gradient(loss, self.model['q_graph'].trainable_weights)
#         gradients, grad_norm = tf.clip_by_global_norm(gradients, self.config.clip_grad_norm)
#
#         with self.config.train_writer.as_default():  # update trainer metrics
#             tf.summary.scalar("grad_norm", grad_norm, self.global_step)
#             tf.summary.scalar("QE_loss", loss, self.global_step)
#             self.global_step.assign_add(1)
#
#         x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
#         y_ = tf.split(y, axis=1, num_or_size_splits=self.config.bptt)[-1]
#         self.fb_input = tf.concat([x_, y_], axis=-1)
#
#         return [gradients, [q_est, s_onehot]]
#
#     def apply_grads(self, gradients):
#         self.optimizer['q_graph'].apply_gradients(zip(gradients, self.model['q_graph'].trainable_weights))
#
#     # def sync_eval_model(self):
#     #     # sync model:
#     #     w = self.model['q_graph'].get_weights()
#     #     self.model['q_graph_eval'].set_weights(w)
#
#     def reset_model_states(self):
#         def reset_recursively(models):
#             for model in models.values():
#                 if isinstance(model, dict):
#                     reset_recursively(model)
#                 else:
#                     model.reset_states()
#
#         reset_recursively(self.model)
#         # if (self.config.channel_name == "ising") or (self.config.channel_name == "trapdoor"):
#         #     self.model['enc'].layers[3].reset_states()
#         #     self.model['enc_eval'].layers[3].reset_states()
#
#     def sync_eval_model(self):
#         # sync DV model:
#         w = self.model['q_graph'].get_weights()
#         self.model['q_graph_test'].set_weights(w)
#
#
# class InputDistInvestigator(object):
#     def __init__(self, model, data, config):
#         """
#         This is a simple routine for the investigation of the learned input distribution
#         """
#         self.config = config
#         self.model = model
#         self.channel_name = config.channel_name
#         self.load_enc_weights()
#         self.data_iterator = data
#         self.bptt = config.bptt
#         self.data_batches = config.data_batches
#         self.feedback = config.feedback
#
#         if self.feedback:
#             self.fb_input = tf.zeros(shape=[config.batch_size, 1, config.x_dim + config.y_dim])
#         else:
#             self.fb_input = tf.zeros(shape=[config.batch_size, 1, config.x_dim])
#
#     def load_enc_weights(self):
#         """
#         set pretrained encoder weights
#         """
#         filepath = os.path.join("./pretrained_models", self.config.channel_name + "_fb", "p_05_best_so_far", "weights")
#         self.model['enc'].load_weights(filepath)
#
#     def train(self):
#         data = self.generate_long_sequence()
#         self.save_sequence(data)
#
#     def generate_long_sequence(self):
#         data = {
#             'x': [],
#             'y': [],
#             'p': [],
#             's': []
#         }
#         for i in range(self.data_batches):
#             [x, y, p, s] = self.model['enc'](self.fb_input)
#
#             x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             y_ = tf.split(y, axis=1, num_or_size_splits=self.config.bptt)[-1]
#
#             if self.feedback:
#                 self.fb_input = tf.concat([x_, y_], axis=-1)
#             else:
#                 self.fb_input = x_
#
#             data['x'].append(x.numpy())
#             data['y'].append(y.numpy())
#             data['p'].append(p.numpy())
#             data['s'].append(s.numpy())
#             print(f"collected {i*self.config.bptt} samples")
#
#         data['x'] = np.concatenate(data['x'], axis=1)[0:14,:,:]
#         data['y'] = np.concatenate(data['y'], axis=1)[0:14,:,:]
#         data['p'] = np.concatenate(data['p'], axis=1)[0:14,:,:]
#         data['s'] = np.concatenate(data['s'], axis=1)[0:14,:,:]
#
#         return data
#
#     def save_sequence(self, data):
#         file_name = "long_sequence_data.mat"
#         savemat(os.path.join(self.config.tensor_board_dir, 'visual',
#                              file_name), data)



