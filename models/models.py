import tensorflow as tf
from models.DV_model import DVModel, DVModel_with_states
from models.enc_model import PMFModel, PMFModelQ_train, PMFModelQ_eval
from models.Q_estimator_model import QModel
from models.mine_sim_models import DVMINEModel, PMFMINEModel, SamplerMINEModel
tf.keras.backend.set_floatx('float64')


def build_model(config):
    if config.model_name == "cap_est":  # for CapEst and CapEstChkpt
        model = {'dv': DVModel(config,
                               bptt=config.bptt,
                               batch_size=config.batch_size,
                               dims=(config.x_dim, config.y_dim)),
                 'dv_eval': DVModel_with_states(config,
                                                bptt=config.bptt,
                                                batch_size=config.batch_size,
                                                dims=(config.x_dim, config.y_dim)),
                 'enc': PMFModel(config),
                 'enc_eval': PMFModel(config)
                 }
    elif config.model_name == "q_est":  # DINE + Enc. models
        model = {  # 'enc': EncModel_PDINE(config),
            'enc': PMFModelQ_train(config),
            'enc_eval': PMFModelQ_eval(config),
            'q_graph': QModel(config, training=True),
            'q_graph_test': QModel(config, training=False)
        }
    # Input Investigation
    elif config.model_name == "input_invest":  # DINE + Enc. models
        model = {
            'enc': PMFModelQ_train(config)
        }
    elif config.model_name == "mine_cap_est":
        model = {
            'dv': DVMINEModel(config),
            'dv_eval': DVMINEModel(config),

            'pmf': PMFMINEModel(config),
            'pmf_eval': PMFMINEModel(config),

            'sampler': SamplerMINEModel(config)
        }
    else:
        raise ValueError("'{}' is an invalid model name")

    return model




