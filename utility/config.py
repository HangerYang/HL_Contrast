import argparse
import json

def get_configs(args):
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
    else:
        raise Exception('config file not defined')
    if args.dataset is None:
        args.dataset = config['dataset']
    if args.epochs is None:
        args.epochs = config['epochs']
    if args.gnn is None:
        args.gnn = config['gnn']
    if args.preepochs is None:
            args.preepochs = config['preepochs']
    if args.loss_type is None:
            args.loss_type = config['loss_type']
    if args.seed is None:
        args.seed = config['seed']
    if args.pre_learning_rate is None:
        args.pre_learning_rate = config['pre_learning_rate']
    if args.learning_rate is None:
        args.learning_rate = config['learning_rate']
    if args.weight_decay is None:
        args.weight_decay = config['weight_decay']
    if args.patience is None:
        args.patience = config['patience']
    if args.hidden_dim is None:
        args.hidden_dim = config['hidden_dim']


    if args.aug_type is None: #adding more aug
        args.aug_type = config['aug_type']
    return args

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='configuration file')
    parser.add_argument('--dataset', help='benchmark dataset : cora, citeseer, pubmed')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train')
    parser.add_argument('--preepochs', type=int, help='Number of epochs to pre-train, only applicable to FBGCN')
    parser.add_argument('--gnn', help='gcn, gat, fbgcn')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--learning_rate', type=float, help='learning rate')
    parser.add_argument('--weight_decay', type=float, help='weight decay')
    parser.add_argument('--patience', type=int, help='patience for early stopping')
    parser.add_argument('--aug_type', help='augmentation type')
    parser.add_argument('--hidden_dim', help='hidden dimension in the model')
    parser.add_argument('--pre_learning_rate', help='pre training learning rate')
    parser.add_argument('--loss_type', help='applying which loss')
    args = parser.parse_args()
    args = get_configs(args)
    return args


    