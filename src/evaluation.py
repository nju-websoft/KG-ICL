import os
import argparse
import torch, logging, sys
import numpy as np
from experiment import Experiment
from tqdm import tqdm
from datetime import datetime


def prepare(args):
    '''preprocess for pretrain'''
    args.gpu = int(args.gpu)
    args.shot = int(args.shot)
    args.attn_dim = int(args.attn_dim)
    if 'without augment' in args.note:
        args.use_augment = False
    if 'without token_set' in args.note:
        args.use_token_set = False
    if 'path' in args.note:
        args.prompt_graph_type = 'path'
    if 'neighbor' in args.note:
        args.prompt_graph_type = 'neighbor'
    if 'path_hop=2' in args.note:
        args.path_hop = 2
    if 'path_hop=3' in args.note:
        args.path_hop = 3
    if 'path_hop=1' in args.note:
        args.path_hop = 1
    def add_note(path, dic):
        for key, value in dic:
            path = path + '_' + key + '_' + str(value)
        return path
    '''set log_path'''
    args.log_path = args.log_path + datetime.now().strftime('%Y%m%d/')
    if not os.path.exists(args.log_path):
        os.mkdir(args.log_path)
    log_note_dict = [('', args.train_dataset_list),
                     ('', args.test_dataset_list),
                     ('dim', args.hidden_dim),
                     ('layer', args.n_layer),
                     ('rel_layer', args.n_relation_encoder_layer),
                     ('', args.note),]
    args.log_path = add_note(args.log_path, log_note_dict)
    '''set save_path'''
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    save_note_dict = [('', args.train_dataset_list),
                        ('', args.train_part),
                        ('dim', args.hidden_dim),
                        ('layer', args.n_layer),
                        ('rel_layer', args.n_relation_encoder_layer),
                        ('', args.note),]
    args.save_path = add_note(args.save_path, save_note_dict)
    '''create dir'''
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    '''set gpu'''
    if torch.cuda.is_available():
        args.device = torch.device("cuda", args.gpu)
    else:
        args.device = torch.device("cpu")

    '''set seed'''
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    '''set logger'''
    logger = logging.getLogger()
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
    console_formatter = logging.Formatter('%(asctime)-8s: %(message)s')
    logging_file_name = args.log_path + '.txt'
    file_handler = logging.FileHandler(logging_file_name)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.formatter = console_formatter
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    args.logger = logger


parser = argparse.ArgumentParser(description="Parser for KGFM")

parser.add_argument('--checkpoint_path', type=str, default='../checkpoint/pretrain/KG-ICL-6L')
parser.add_argument('--train_dataset_list', type=str, default=None)
parser.add_argument('--valid_dataset_list', type=str, default=None)
parser.add_argument('--test_dataset_list', type=str, default='WN18RR_v1_ind')
parser.add_argument('--train_part', type=str, default=None)
parser.add_argument('--valid_part', type=str, default=None)
parser.add_argument('--test_part', type=str, default='test')

# model structure
parser.add_argument('--use_attn', type=bool, default=True)  # False
parser.add_argument('--attn_type', type=str, default='Sigmoid', help='GAT or Sigmoid')  #
parser.add_argument('--AGG', type=str, default='max')  # sum
parser.add_argument('--AGG_rel', type=str, default='max')  # sum
parser.add_argument('--MSG', type=str, default='concat')  # add, concat, mix
parser.add_argument('--use_augment', type=bool, default=False)
parser.add_argument('--use_token_set', type=bool, default=True)
parser.add_argument('--use_prompt_graph', type=bool, default=True)
parser.add_argument('--prompt_graph_type', type=str, default='all', choices=['all', 'path', 'neighbor'])
parser.add_argument('--path_hop', type=int, default=3)  # 3

# hyper_parameters
parser.add_argument('--shot', default=5)
parser.add_argument('--hidden_dim', type=int, default=32)
parser.add_argument('--attn_dim', type=int, default=5)
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--n_relation_encoder_layer', type=int, default=3)
parser.add_argument('--n_layer', type=int, default=6)
parser.add_argument('--train_batch_size', type=int, default=75)
parser.add_argument('--test_batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lamb', type=float, default=0.0001)
parser.add_argument('--decay_rate', type=float, default=0.995)
parser.add_argument('--act', type=str, default='idd')
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--relation_mask_rate', type=float, default=0.0)
parser.add_argument('--adversarial_temperature', type=float, default=0.5)
parser.add_argument('--finetune', type=bool, default=False)

# others
parser.add_argument('--data_path', type=str, default='../processed_data/')
parser.add_argument('--log_path', type=str, default='../log/pretrain/')
parser.add_argument('--save_path', type=str, default='../checkpoint/pretrain/')
parser.add_argument('--seed', type=str, default=1)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--use_rspmm', type=bool, default=False)
parser.add_argument('--note', type=str, default='')

args = parser.parse_args()

if not os.path.exists(args.log_path):
    os.makedirs(args.log_path)

'''preprocess dataset list'''
if args.train_dataset_list is None:
    pass
elif len(args.train_dataset_list.split()) > 1:
    args.train_dataset_list = args.train_dataset_list.split()
else:
    args.train_dataset_list = [args.train_dataset_list]
if args.valid_dataset_list is None:
    pass
elif len(args.valid_dataset_list.split()) > 1:
    args.valid_dataset_list = args.valid_dataset_list.split()
else:
    args.valid_dataset_list = [args.valid_dataset_list]
if args.test_dataset_list is None:
    pass
elif len(args.test_dataset_list.split()) > 1:
    args.test_dataset_list = args.test_dataset_list.split()
else:
    args.test_dataset_list = [args.test_dataset_list]

args.train_dirs = [os.path.join(args.data_path, dataset, args.train_part) for dataset in args.train_dataset_list] if args.train_dataset_list is not None else None
args.valid_dirs = [os.path.join(args.data_path, dataset, args.valid_part) for dataset in args.valid_dataset_list] if args.valid_dataset_list is not None else None
args.test_dirs = [os.path.join(args.data_path, dataset, args.test_part) for dataset in args.test_dataset_list] if args.test_dataset_list is not None else None


prepare(args)


def test(args, experiment):
    '''test all'''
    experiment.load_test()
    mrr, out_str = experiment.evaluate(mode='test')
    args.logger.info(out_str)


if __name__ == '__main__':
    args.logger.info(args)
    experiment = Experiment(args)
    experiment.load_checkpoint(args.checkpoint_path, gpu=args.gpu)
    test(args, experiment)


