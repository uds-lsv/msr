import yaml
import os
import torch.nn.functional as F
import logging
import json
import pickle
import datetime
from torch.utils.tensorboard import SummaryWriter
from trainers.vanilla_trainer import VanillaTrainer
from trainers.msr_trainer import MSRTrainer
from trainers.msr_trainer_ner import MSRNERTrainer

def create_trainer(args, logger, log_dir, random_state):
    if args.trainer_name == 'vanilla':
        trainer = VanillaTrainer(args, logger, log_dir, random_state)
    elif args.trainer_name == 'msr':  # some other variations of meta_cleaning method
        trainer = MSRTrainer(args, logger, log_dir, random_state)
    elif args.trainer_name == 'msr_ner':  # some other variations of meta_cleaning method
        trainer = MSRNERTrainer(args, logger, log_dir, random_state)
    else:
        raise NotImplementedError('Unknown Trainer Name')

    return trainer




def sanity_checks(args):
    if args.dataset in ['Yoruba', 'Hausa']:
        assert args.model_name in ['bert-base-multilingual-cased']

    if args.dataset in ['TREC']:
        assert args.model_name in ['roberta-base']




def load_config(args):
    model_config = {}
    model_config['drop_rate'] = args.bert_dropout_rate
    return model_config

def save_config(save_dir, config_name, args):
    argparse_dict = vars(args)
    save_path = os.path.join(save_dir, f'{config_name}.yaml')

    with open(save_path, 'w') as file:
        yaml.dump(argparse_dict, file)

def pickle_save(file_name, data):
    with open(file_name, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def pickle_load(file_name):
    with open(file_name, 'rb') as handle:
        b = pickle.load(handle)
    return b



def entropy_loss(ul_y):
    p = F.softmax(ul_y, dim=1)
    return -(p * F.log_softmax(ul_y, dim=1)).sum(dim=1).mean(dim=0)


def create_log_path(log_root, args, starting_time):
    staring_time_str = starting_time.strftime("%m_%d_%H_%M_%S")
    suffix = staring_time_str

    suffix += f'_{args.trainer_name}'
    suffix += f'_lbs{args.nl_batch_size}'

    log_dir = os.path.join(log_root, suffix)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, 'log.txt')
    return log_path, log_dir


def create_logger(log_root, args):
    starting_time = datetime.datetime.now()

    log_path, log_dir = create_log_path(log_root, args, starting_time)

    # check if the file exist

    console_logging_format = "%(levelname)s %(message)s"
    file_logging_format = "%(levelname)s: %(asctime)s: %(message)s"

    # configure logger
    logging.basicConfig(level=logging.INFO, format=console_logging_format)
    logger = logging.getLogger()

    # create a file handler for output file
    handler = logging.FileHandler(log_path)

    # set the logging level for log file
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter(file_logging_format)
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    return logger, log_dir


def save_args(log_dir, args):
    arg_save_path = os.path.join(log_dir, 'config.json')
    with open(arg_save_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)


def create_summary_writers(sw_dir, train_summary_name='train',
                           validation_summary_name='validation',
                           test_summary_name='test'):
    train_sw_path = os.path.join(sw_dir, train_summary_name)
    validation_sw_path = os.path.join(sw_dir, validation_summary_name)
    test_sw_path = os.path.join(sw_dir, test_summary_name)

    for p in [train_sw_path, validation_sw_path, test_sw_path]:
        if not os.path.exists(p):
            os.makedirs(p)

    sw_train = SummaryWriter(log_dir=train_sw_path)
    sw_val = SummaryWriter(log_dir=validation_sw_path)
    sw_test = SummaryWriter(log_dir=test_sw_path)
    return sw_train, sw_val, sw_test
