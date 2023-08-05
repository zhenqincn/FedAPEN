import argparse
import torch
import numpy as np

from data_loader import MyDataLoader
from recorder import Recorder
from tools.utils import *
from node import ServerNode, Node
from trainer import Trainer
import random
from datetime import date


def set_seed(seed):
    """
    set random seed
    @param seed:
    @return:
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if seed == -1:
        torch.backends.cudnn.deterministic = False
    else:
        torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    today = date.today()
    
    parser = argparse.ArgumentParser(description='FedDML')
    # FL settings
    parser.add_argument('--nodes', default=10, type=int, help='number of clients')
    parser.add_argument('-k', default=10, type=int, help='the number of clients to be selected for updating in a round')
    parser.add_argument('--rounds', '-r', default=100, type=int, help='number of rounds for federated averaging')
    parser.add_argument('--local_epoch', '-l', default=5, type=int, help='the number of local epochs before share the local updates')
    
    # Model settings
    parser.add_argument('--private_model', default='cnn2_bn', type=str)
    parser.add_argument('--shared_model', default='cnn2_bn', type=str)
    parser.add_argument('--init', default='kaiming', type=str, help='init function, "none" means no init')

    # Optimizer settings
    parser.add_argument('--lr', default=1e-2, type=float, help='learning rate of local training')
    parser.add_argument('--lr_decay', default=0.99, type=float, help='learning rate decay')
    parser.add_argument('--optimizer', default='sgd', type=str, help='optimizer for client, optional in sgd, adam')
    parser.add_argument('--algorithm', default='learned_adaptive_training', type=str, help='optional in ["learned_adaptive_training", "equal_training", "learned_adaptive", "fed_avg", "individual", "fed_avg_tune"]')
    parser.add_argument('--he', default=0, type=int, help='whether enable the heterogeneity of private models')

    # Data settings
    parser.add_argument('--dataset', '-d', default='cifar-10', type=str, help='the name of dataset')
    parser.add_argument('--download', default=False, action='store_true', help='whether download the corresponding dataset if it does not exists')
    parser.add_argument('--num_classes', default=100, type=int, help='num classes')
    parser.add_argument('--iid', default="0", choice=['0', '2', '4', '6', '8', 'dir0.05', 'dir0.1', 'dir0.5', 'dir1.0', 'dir5.0'], type=str, help='the type of data non-iid, number means pathological non-IID with alpha=iid / 10, note that 0 means iid (alpha=1.0), if set to "dir{iid}", meaning practical non-iid with alpha=iid')
    parser.add_argument('--ratio_train_adaptability', default='[0.95, 0.05]', type=str, help='the ratio of training set, validation set')
    parser.add_argument('--batch', '-b', default=64, type=int, help='batch size of local training')

    # Other settings
    parser.add_argument('--gpu', '-g', default=0, type=int, help='id of gpu')
    parser.add_argument('--model_save', default='none', type=str, help='save mode of model, optional in ["none", "best", "verbose"]')
    parser.add_argument('--log_root', default='logs', type=str, help='the root path of log directory')
    parser.add_argument('--seed', default=int(str(today.year) + str(today.month) + str(today.day)), type=int, help='the seed of random split dataset')

    args = parser.parse_args()

    # these three approaches require adaptability set
    if args.algorithm not in ["learned_adaptive_training", "equal_training", "learned_adaptive"]:
        args.ratio_train_adaptability = '[1.0, 0.0]'
    
    if args.dataset.lower() == 'cifar-100':
        args.num_classes = 100
    else:
        args.num_classes = 10
    
    set_seed(args.seed)
    dl = MyDataLoader(args)

    recorder = Recorder(args)
    trainer = Trainer(recorder, args)

    # initialize the server node of FL
    server_node = ServerNode(dl.eval_loader_complete, recorder=recorder, args=args)
    # initialize client nodes of FL
    node_list = [Node(idx, train_loader=dl.train_loader_list[idx], adaptability_loader=dl.adaptability_loader_list[idx], eval_loader=dl.eval_loader_list[idx], args=args) for idx in range(args.nodes)]
    recorder.register_node_list(node_list)
    server_node.aggregate(node_list, cur_round=-1)  # first initialize all the shared model at the same state
    
    print('Algorithm: {0}'.format(args.algorithm))
    for cur_round in range(args.rounds):
        print('\n===============The {:d}-th round==============='.format(cur_round + 1))
        lr_schedule_exponential(cur_round, node_list, args=args)
            
        if args.algorithm.lower() != 'individual':
            for node in node_list:
                node.refresh_optimizer()
            
        for node in node_list:
            trainer(node, cur_round)  # local training
            
        if args.algorithm.lower() in ['learned_adaptive_training', 'equal_training', 'learned_adaptive', 'heuristic_mutual', 'mutual', 'fed_avg', 'apfl']:   # first aggregate, then eval
            server_node.aggregate(node_list, cur_round)
            for node in node_list:
                # update the weight for inference according to the validation set of the corresponding node
                if args.algorithm.lower() in ['learned_adaptive_training', 'learned_adaptive', 'equal_training']:
                    node.learn_weight_for_inference()
                # the server will unify the weight of private model for training
                recorder.eval(node)
                
        elif args.algorithm.lower() == 'individual':    # for individual, eval, without aggregation
            for node in node_list:
                recorder.eval(node)
                
        elif args.algorithm.lower() == 'fed_avg_tune':  # for fed_avg_tune, first eval, then aggregate
            for node in node_list:
                recorder.eval(node)
            server_node.aggregate(node_list, cur_round)
        recorder.summary(cur_round)
