import tensorflow as tf
from models.DV_model import DVModel, DVModel_with_states
from models.enc_model import EncModel, EncModel_no_p, EncModel_PDINE, EncModel_no_p_new
from models.Q_estimator_model import Q_model_noise
tf.keras.backend.set_floatx('float64')


def build_model(config):
    # Capacity estimation models
    if config.model_name == "di_with_enc_states":  # DINE + Enc. models
        if config.with_p == 1:
            model = {'dv': DVModel(config,
                                   bptt=config.bptt,
                                   batch_size=config.batch_size,
                                   dims=(config.x_dim, config.y_dim)),
                     'dv_eval': DVModel_with_states(config,
                                        bptt=config.bptt,
                                        batch_size=config.batch_size,
                                        dims=(config.x_dim, config.y_dim)),
                     'enc': EncModel(config),
                     'enc_eval': EncModel(config)
                     }
        else:
            model = {'dv': DVModel(config,
                                   bptt=config.bptt,
                                   batch_size=config.batch_size,
                                   dims=(config.x_dim, config.y_dim)),
                     'dv_eval': DVModel_with_states(config,
                                                    bptt=config.bptt,
                                                    batch_size=config.batch_size,
                                                    dims=(config.x_dim, config.y_dim)),
                     'enc': EncModel_no_p(config),
                     'enc_eval': EncModel_no_p(config)
                     }
    # Q graph model
    elif config.model_name == "q_est":  # DINE + Enc. models
        model = {#'enc': EncModel_PDINE(config),
                 'enc': EncModel_no_p_new(config),
                 'enc_eval': EncModel_PDINE(config),
                 'q_graph': Q_model_noise(config, training=True),
                 'q_graph_test': Q_model_noise(config, training=False)
                 }
    # Input Investigation
    elif config.model_name == "input_invest":  # DINE + Enc. models
        model = {
            'enc': EncModel_no_p_new(config)
        }
    else:
        raise ValueError("'{}' is an invalid model name")

    return model

    # OLD MODELS - investigate further
    # new pdine model
    # elif config.model_name == "pdine_new":
    #     model = {'dv': DVModel_pdine(config, mode='train'),
    #              'dv_eval': DVModel_pdine(config, mode='eval'),
    #              'enc': EncModel_pdine(config),
    #              'enc_eval': EncModel_pdine(config)}
    #
    # #### OLD models:
    # # first modification of di_with_enc_states, got too complicated
    # elif config.model_name == "PDINE":  # DINE + Enc. models
    #     model = {'dv': DVModel(config,
    #                            bptt=config.bptt,
    #                            batch_size=config.batch_size,
    #                            dims=(config.x_dim, config.y_dim)),
    #              'dv_eval': DVModel_with_states(config,
    #                                             bptt=config.bptt,
    #                                             batch_size=config.batch_size,
    #                                             dims=(config.x_dim, config.y_dim)),
    #              'enc': EncModel_PDINE(config),
    #              'enc_eval': EncModel_PDINE(config),
    #              'q_graph': Q_model(config)
    #              }
    # # a slightly modified version of di_with_enc_states (preferably use the older)
    # elif config.model_name == "di_with_enc_states_new":  # DINE + Enc. models
    #     model = {'dv': DVModel(config,
    #                            bptt=config.bptt,
    #                            batch_size=config.batch_size,
    #                            dims=(config.x_dim, config.y_dim)),
    #              'dv_eval': DVModel_with_states(config,
    #                                             bptt=config.bptt,
    #                                             batch_size=config.batch_size,
    #                                             dims=(config.x_dim, config.y_dim)),
    #              'enc': EncModel_no_p_new(config),
    #              'enc_eval': EncModel_no_p_new(config)
    #              }
    # # oldest dine+enc
    # elif config.model_name == "di_with_enc":  # DINE + Enc. models
    #     model = {'dv': DVModel(config,
    #                            bptt=config.bptt,
    #                            batch_size=config.batch_size,
    #                            dims=(config.x_dim, config.y_dim)),
    #              'dv_eval': DVModel(config,
    #                                 bptt=config.bptt,
    #                                 batch_size=config.batch_size,
    #                                 dims=(config.x_dim, config.y_dim)),
    #              'enc': EncModel(config),
    #              'enc_eval': EncModel(config)
    #              }
    # # only di est
    # elif config.model_name == "di":  # only DINE model
    #     model = {'dv': DVModel(config,
    #                            bptt=config.bptt,
    #                            batch_size=config.batch_size,
    #                            # names=('y', 'xy'),
    #                            dims=(config.x_dim, config.y_dim)),
    #              'dv_eval': DVModel(config,
    #                            bptt=config.bptt,
    #                            batch_size=config.batch_size,
    #                            # names=('y', 'xy'),
    #                            dims=(config.x_dim, config.y_dim))
    #              }
    # elif config.model_name == "pdine_gen_alphabet":  # DINE + Enc. models
    #     if config.with_p == 1:
    #         model = {'dv': DVModel(config,
    #                                bptt=config.bptt,
    #                                batch_size=config.batch_size,
    #                                dims=(config.x_dim, config.y_dim)),
    #                  'dv_eval': DVModel_with_states(config,
    #                                     bptt=config.bptt,
    #                                     batch_size=config.batch_size,
    #                                     dims=(config.x_dim, config.y_dim)),
    #                  'enc': EncModel(config),
    #                  'enc_eval': EncModel(config)
    #                  }
    #     else:
    #         model = {'dv': DVModel(config,
    #                                bptt=config.bptt,
    #                                batch_size=config.batch_size,
    #                                dims=(config.x_dim, config.y_dim)),
    #                  'dv_eval': DVModel_with_states(config,
    #                                                 bptt=config.bptt,
    #                                                 batch_size=config.batch_size,
    #                                                 dims=(config.x_dim, config.y_dim)),
    #                  'enc': EncModel_gen_alphabet(config),
    #                  'enc_eval': EncModel_gen_alphabet(config)
    #                  }
    # elif config.model_name == "PDINE_no_q":  # DINE + Enc. models
    #     model = {'dv': DVModel(config,
    #                            bptt=config.bptt,
    #                            batch_size=config.batch_size,
    #                            dims=(config.x_dim, config.y_dim)),
    #              'dv_eval': DVModel(config,
    #                             bptt=config.bptt,
    #                             batch_size=config.batch_size,
    #                             dims=(config.x_dim, config.y_dim)),
    #              'enc': EncModel_PDINE(config),
    #              'enc_eval': EncModel_PDINE(config)
    #              }
    # elif config.model_name == "cont_ndt":  # DINE + Enc. models
    #     model = {'dv': DVModel(config,
    #                            bptt=config.bptt,
    #                            batch_size=config.batch_size,
    #                            dims=(config.x_dim, config.y_dim)),
    #              'dv_eval': DVModel(config,
    #                             bptt=config.bptt,
    #                             batch_size=config.batch_size,
    #                             dims=(config.x_dim, config.y_dim)),
    #              'enc': NDTModel(config),
    #              'enc_eval': NDTModel(config)
    #              }




