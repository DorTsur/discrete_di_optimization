import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
import logging


from trainers.capacity_estimation_di import CapEstDI, CapEstDIChkpt
from trainers.input_investigation import InputDistInvestigator
from trainers.q_graph_trainer import QgraphTrainer
from trainers.mine_trainer import CapEstMINE


logger = logging.getLogger("logger")

def build_trainer(model, data, config):
    if config.trainer_name == "cap_est":
        trainer = CapEstDI(model, data, config)  # DINE + Encoder (currently binary alphabets)
    elif config.trainer_name == "q_est":
        trainer = QgraphTrainer(model, data, config)
    elif config.trainer_name == "input_invest":
        trainer = InputDistInvestigator(model, data, config)
    elif config.trainer_name == "cap_est_checkpoint":
        trainer = CapEstDIChkpt(model, data, config)
    elif config.trainer_name == "mine_pmf":
        trainer = CapEstMINE(model, data, config)  # DINE + Encoder (currently binary alphabets)
    else:
        raise ValueError("'{}' is an invalid trainer name")
    return trainer


######################






