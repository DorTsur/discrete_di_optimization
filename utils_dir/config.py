from bunch import Bunch
from collections import OrderedDict
import json
import logging
from pathlib import Path
from random import randint

CONFIG_WAIVER = ['save_model', 'tracking_uri', 'quiet', 'sim_dir', 'train_writer', 'test_writer', 'valid_writer']
MAX_SEED = 1000000
logger = logging.getLogger("logger")

class Config(Bunch):
    """ class for handling dicrionary as class attributes """

    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)

    def print(self):
        line_width = 132
        line = "-" * line_width
        logger.info(line + "\n" +
              "| {:^35s} | {:^90} |\n".format('Feature', 'Value') +
              "=" * line_width)
        for key, val in sorted(self.items(), key= lambda x: x[0]):
            if isinstance(val, OrderedDict):
                raise NotImplementedError("Nested configs are not implemented")
            else:
                logger.info("| {:35s} | {:90} |\n".format(key, str(val)) + line)
        logger.info("\n")

def read_json_to_dict(fname):
    """ read json config file into ordered-dict """
    fname = Path(fname)
    with fname.open('rt') as handle:
        config_dict = json.load(handle, object_hook=OrderedDict)
        return config_dict

def read_config(args):
    """ read config from json file and update by the command line arguments """
    if args.config is not None:
        json_file = args.config
    else:
        # raise ValueError("preprocess config: config path wasn't specified")
        json_file = './configs/capacity_estimation.json'  # current pdine config
        # json_file = './configs/capacity_estimation_from_checkpoint.json'  # current pdine config
        # json_file = './configs/q_graph.json'  # current pdine config
        # json_file = './configs/input_investigation.json'  # input investigation config


        # json_file = './configs/pdine_new_debug.json'  # for new pdine config
        # json_file = './configs/pdine_old_for_debug.json'  # for old pdine config
        # json_file = './configs/continuous_capacity_dine.json'  # for old pdine config

    config_dict = read_json_to_dict(json_file)
    config = Config(config_dict)

    for key in sorted(vars(args)):
        val = getattr(args, key)
        if val is not None:
            setattr(config, key, val)

    if args.seed is None and config.seed is None:
        config.seed = randint(0, MAX_SEED)

    return config
