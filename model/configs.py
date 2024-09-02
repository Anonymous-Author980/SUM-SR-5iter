# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pprint
import os
import shutil


save_dir = Path('../../../gluster/exp-notgan-pretrain-5iter')

def str2bool(v):
    """string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.set_dataset_dir(self.video_type, self.exp, self.summary_rate, self.iter)

    def set_dataset_dir(self, video_type='TVSum', exp=0, reg=0.15, iter=0):
        self.log_dir = save_dir.joinpath('reg'+str(reg), 'exp'+str(exp), 'iter'+str(iter), video_type, 'logs/split'+str(self.split_index))
        self.score_dir = save_dir.joinpath('reg'+str(reg), 'exp'+str(exp), 'iter'+str(iter), video_type, 'results/split'+str(self.split_index))
        self.save_dir = save_dir.joinpath('reg'+str(reg), 'exp'+str(exp), 'iter'+str(iter), video_type, 'models/split'+str(self.split_index))
        self.pretrain_save_dir = save_dir.joinpath('reg'+str(reg), 'exp'+str(exp), 'iter'+str(iter), video_type, 'models/split'+str(self.split_index) + '/pretrain')
        self.seed_dir = save_dir.joinpath('reg'+str(reg), 'exp'+str(exp), 'iter'+str(iter), video_type, 'seed/split' + str(self.split_index))
        if iter > 0:
            self.prev_dir = save_dir.joinpath('reg'+str(reg), 'exp'+str(exp), 'iter'+str(iter-1), video_type)
        
        #if directory not exist, make directories
        logExist = os.path.exists(self.log_dir)
        scoreExist = os.path.exists(self.score_dir)
        saveExist = os.path.exists(self.save_dir)
        seedExist = os.path.exists(self.seed_dir)
        pretrainExist = os.path.exists(self.pretrain_save_dir)
        
        if not logExist:
            os.makedirs(self.log_dir)
            print("Successfully make log_dir")
        else:
            shutil.rmtree(self.log_dir, ignore_errors=True)
            os.makedirs(self.log_dir)
            print("Successfully remake log_dir")
        if not scoreExist:
            os.makedirs(self.score_dir)
            print("Successfully make score_dir")
        if not saveExist:
            os.makedirs(self.save_dir)
            print("Successfully make save_dir")
        
        if not seedExist:
            os.makedirs(self.seed_dir)
            print("Successfully make seed_dir")
        if not pretrainExist:
            os.makedirs(self.pretrain_save_dir)
            print("Successfully make pretrain_save_dir")
        
    def __repr__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str

def get_config(parse=True, **optional_kwargs):
    """
    Get configurations as attributes of class
    1. Parse configurations with argparse.
    2. Create Config class initilized with parsed kwargs.
    3. Return Config class.
    """
    parser = argparse.ArgumentParser()

    # Mode
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--full_mode', type=str, default='all')
    parser.add_argument('--verbose', type=str2bool, default='true')
    parser.add_argument('--video_type', type=str, default='TVSum')

    # Model
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--summary_rate', type=float, default=0.7)

    # Train
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--recon_n_epochs', type=int, default=100)
    parser.add_argument('--clip', type=float, default=5.0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--autoencoder_lr', type=float, default=1e-4)
    parser.add_argument('--split_index', type=int, default=0)
    parser.add_argument('--autoencoder', type=str, default='AAE')
    parser.add_argument('--exp', type=int, default=0)
    parser.add_argument('--iter', type=int, default=0)
    #parser.add_argument('--autoencoder_slow_start', type=int, default=15)

    
    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)


if __name__ == '__main__':
    config = get_config()
    import ipdb
    ipdb.set_trace()