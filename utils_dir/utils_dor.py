import argparse
import logging
import os
import glob
import shutil
import sys
import tensorflow as tf
from utils_dir.config import read_config
from utils_dir.logger import set_logger_and_tracker

CONFIG_WAIVER = ['save_model', 'quiet', 'sim_dir']

logger = logging.getLogger("logger")

# Obtaining simulation arguments
def get_args():
    """" collects command line arguments """

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--config',                  default=None, type=str,   help='configuration file')
    argparser.add_argument('--exp_name',                default=None, type=str,   help='experiment name')
    argparser.add_argument('--run_name',                default=None, type=str,   help='run name')
    argparser.add_argument('--model_name',              default=None, type=str,   help='run name')
    argparser.add_argument('--trainer_name',            default=None, type=str,   help='run name')
    argparser.add_argument('--tag_name',                default=None, type=str,   help='tag name')
    argparser.add_argument('--contrastive_duplicates',  default=None, type=int,   help='contrastive duplicates')
    argparser.add_argument('--data_path',               default=None, type=str,   help='data path')
    argparser.add_argument('--data_version',            default=None, type=str,   help='data version')
    argparser.add_argument('--batch_size',              default=None, type=int,   help='batch size in training')
    argparser.add_argument('--bptt',                    default=None, type=int,   help='backprop truncation')
    argparser.add_argument('--x_dim',                   default=None, type=int,   help='x dimension')
    argparser.add_argument('--y_dim',                   default=None, type=int,   help='y dimension')
    argparser.add_argument('--seed',                    default=None, type=int,   help='randomization seed')
    argparser.add_argument('--GE_b',                    default=None, type=float, help='for Gilbert-Elliot')
    argparser.add_argument('--GE_g',                    default=None, type=float, help='for Gilbert-Elliot')
    argparser.add_argument('--GE_p_b',                  default=None, type=float, help='for Gilbert-Elliot')
    argparser.add_argument('--GE_p_g',                  default=None, type=float, help='for Gilbert-Elliot')
    argparser.add_argument('--p_bsc',                   default=None, type=float, help='for BSC')
    argparser.add_argument('--p_z',                     default=None, type=float, help='for Z-channel')
    argparser.add_argument('--p_s',                     default=None, type=float, help='for S-channel')
    argparser.add_argument('--lr',                      default=None, type=float, help='learning rate')
    argparser.add_argument('--num_epochs',              default=None, type=int, help='number of epochs')
    argparser.add_argument('--feedback',                default=None, type=int, help='feedbask indicator')
    argparser.add_argument('--channel_name',            default=None, type=str, help='channel scenario')
    argparser.add_argument('--T',                       default=None, type=int, help='return approximation length')
    argparser.add_argument('--optimizer',               default=None, type=str, help='optimizer choice')
    argparser.add_argument('--lr_SGD',                  default=None, type=float, help='optimizer choice')
    argparser.add_argument('--batches',                 default=None, type=int, help='amount of batches')
    argparser.add_argument('--decay',                   default=None, type=int, help='choice of lrdecay use for training')
    argparser.add_argument('--p_ising',                 default=None, type=float, help='ising channel parameter')
    argparser.add_argument('--p_trapdoor',              default=None, type=float, help='trapdoor channel parameter')
    argparser.add_argument('--p_post',                  default=None, type=float, help='POST channel parameter p(y|x,s)')
    argparser.add_argument('--eta_post',                default=None, type=float, help='noisy POST channel parameter p(s|y)')
    argparser.add_argument('--reset_channel',           default=None, type=int, help='for me - did i reset channel')
    argparser.add_argument('--clip_grad_norm_enc',      default=None, type=float, help='the values of encoder clipping')
    argparser.add_argument('--clip_grad_norm',          default=None, type=float, help='the values of dv grad clipping')


    argparser.add_argument('--quiet',                   dest='quiet', action='store_true')
    argparser.set_defaults(quiet=False)

    args = argparser.parse_args()
    return args

# GPU usage function
def gpu_init():
    """ Allows GPU memory growth """

    gpus = tf.config.experimental.list_physical_devices('GPU')
    logger.info("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            logger.info(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            logger.info("MESSAGE", e)

# Saving simulation scripts function
def save_scripts(config):
    path = os.path.join(config.tensor_board_dir, 'scripts')
    if not os.path.exists(path):
        os.makedirs(path)
    scripts_to_save = glob.glob('./**/*.py', recursive=True) + [config.config]
    if scripts_to_save is not None:
        for script in scripts_to_save:
            dst_file = os.path.join(path, os.path.basename(script))
            shutil.copyfile(os.path.join(os.path.dirname(sys.argv[0]),script), dst_file)

# pre-processing function
def preprocess_meta_data():
    args = get_args()
    config = read_config(args)
    gpu_init()
    set_logger_and_tracker(config)
    save_scripts(config)
    return config