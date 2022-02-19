import datetime
import logging
import os
import tensorflow as tf
logger = logging.getLogger("logger")

def set_logger(config):
    ''' define logger object to log into file and to stdout'''

    logFormatter = logging.Formatter("%(message)s")
    logger_ = logging.getLogger("logger")
    log_path = os.path.join(config.tensor_board_dir, "logger.log")

    logger.setLevel(logging.INFO)

    fileHandler = logging.FileHandler(log_path)
    fileHandler.setFormatter(logFormatter)
    logger_.addHandler(fileHandler)

    if not config.quiet:
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        logger_.addHandler(consoleHandler)

def set_logger_and_tracker(config):
    # create tensorboard directories:
    config.tensor_board_dir = os.path.join('.',
                                           'results',
                                           config.exp_name,
                                           config.data_name,
                                           config.run_name,
                                           config.tag_name,
                                           "{}_{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                                          config.seed))
    if not os.path.exists(config.tensor_board_dir):
        os.makedirs(config.tensor_board_dir)

    path = os.path.join(config.tensor_board_dir, 'visual')
    if not os.path.exists(path):
        os.makedirs(path)

    train_log_dir = os.path.join(config.tensor_board_dir, 'train')
    config.train_writer = tf.summary.create_file_writer(train_log_dir)

    test_log_dir =  os.path.join(config.tensor_board_dir, 'test')
    config.test_writer = tf.summary.create_file_writer(test_log_dir)

    set_logger(config)
