import numpy as np
from tensorflow.keras.optimizers.schedules import ExponentialDecay


def exp_dec_lr(config, data, model):
    if config.optimizer == "SGD":
        init_lr = config.lr_SGD
    else:
        init_lr = config.lr
    total_iterations_num = config.num_epochs * len([_ for _ in data["train"]()])
    decay_steps = np.floor(total_iterations_num / (np.log(0.1)/np.log(0.99)))
    # log(0.1) for one order of magnitude, log(0.01) for two orders of magnitude
    return ExponentialDecay(initial_learning_rate=init_lr,
                                        decay_steps=decay_steps,
                                        decay_rate=0.995,
                                        staircase=True)

def cyclic_lr(config, data):
    if config.optimizer == "SGD":
        init_lr = config.lr_SGD
    else:
        init_lr = config.lr
    num_cycles = 2
    total_iterations_num = config.num_epochs * len([_ for _ in data["train"]()])
    cycle_len = total_iterations_num/num_cycles
