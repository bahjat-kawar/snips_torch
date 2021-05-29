import argparse
import traceback
import time
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.utils.tensorboard as tb
import copy
from runners import *

import os

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--config', type=str, required=True,  help='Path to the config file')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--exp', type=str, default='exp', help='Path for saving running related data.')
    parser.add_argument('--doc', type=str, required=True, help='A string for documentation purpose. '
                                                               'Will be the name of the log folder.')
    parser.add_argument('--comment', type=str, default='', help='A string for experiment comment')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('-i', '--image_folder', type=str, default='images', help="The folder name of samples")

    parser.add_argument('-n', '--num_variations', type=int, default=1, help='Number of variations to produce')
    parser.add_argument('-s', '--sigma_0', type=float, default=0.1, help='Noise std to add to observation')
    parser.add_argument('--degradation', type=str, default='sr4', help='Degradation: inp | deblur_uni | deblur_gauss | sr2 | sr4 | cs4 | cs8 | cs16')

    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, 'logs', args.doc)

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.load(f)
    new_config = dict2namespace(config)

    tb_path = os.path.join(args.exp, 'tensorboard', args.doc)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    os.makedirs(os.path.join(args.exp, 'image_samples'), exist_ok=True)
    args.image_folder = os.path.join(args.exp, 'image_samples', args.image_folder)
    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)
    else:
        response = input("Image folder already exists. Overwrite? (Y/N)")
        if response.upper() == 'Y':
            overwrite = True

        if overwrite:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        else:
            print("Output image folder exists. Program halted.")
            sys.exit(0)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))
    logging.info("Config =")
    print(">" * 80)
    config_dict = copy.copy(vars(config))
    print(yaml.dump(config_dict, default_flow_style=False))
    print("<" * 80)

    try:
        runner = NCSNRunner(args, config)
        runner.sample()
    except:
        logging.error(traceback.format_exc())

    return 0


if __name__ == '__main__':
    sys.exit(main())
