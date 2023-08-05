import os
from copy import deepcopy

import torch.nn as nn
import torch.optim

from models.model_helper import get_model
from tools.nn_utils import *

CE_Loss = nn.CrossEntropyLoss()


def init_optimizer(model, args) -> torch.optim.Optimizer:
    optimizer = []
    if args.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    return optimizer


class ServerNode(object):
    def __init__(self, eval_loader, recorder, args):
        self.args = args
        self.shared_model = get_model(args.shared_model, args.num_classes)
        init_weights(self.shared_model, args.init)
        self.device = torch.device('cuda:{0}'.format(args.gpu))
        self.eval_loader = eval_loader
        self.staged_non_iid_degree = []
        self.actual_private_weight_training = []
        self.actual_private_weight_inference = []
        self.recorder = recorder
        self.recorder.register_server_node(self)

    def aggregate(self, node_list: list, cur_round):
        if cur_round == -1:
            for node in node_list:
                node.shared_model.load_state_dict(self.shared_model.state_dict())
        else:
            weights = [node.shared_model.state_dict() for node in node_list]

            # global aggregation
            weights_avg = deepcopy(weights[0])
            for key in weights_avg.keys():
                for i in range(1, len(weights)):
                    weights_avg[key] += weights[i][key].detach()
                weights_avg[key] = torch.div(weights_avg[key], len(weights))
            self.shared_model.load_state_dict(weights_avg)

            for node in node_list:
                node.shared_model.load_state_dict(weights_avg)

            if self.recorder is not None and self.recorder.args.model_save == 'verbose':
                torch.save(weights_avg, os.path.join(self.args.log_root, str(self.recorder.time_stamp), 'models', 'globalmodel_round{0}_{1}.pt'.format(cur_round, node.args.shared_model)))


class Node(object):
    def __init__(self, idx, train_loader, adaptability_loader, eval_loader, args):
        self.alpha_apfl = 0.01
        self.idx = idx
        self.args = args
        self.staged_learned_weight_inference = []  # save the learned weight for inference in each round
        self.device = torch.device('cuda:{0}'.format(args.gpu))
        # these four algorithms support heterogeneous models
        if self.args.algorithm in ['learned_adaptive_training', 'equal_training', 'learned_adaptive'] and self.args.he == 1:
            print('Node{0}, heterogeneous'.format(self.idx))
            self.private_model = get_model(args.private_model, num_classes=args.num_classes, client_id=self.idx)
        else:
            print('Node{0}, homogeneous'.format(self.idx))
            self.private_model = get_model(args.private_model, num_classes=args.num_classes, client_id=None)
        init_weights(self.private_model, args.init)
        print('Client {0}, Number of parameters: '.format(self.idx), sum(p.numel() for p in self.private_model.parameters() if p.requires_grad))
        self.private_optimizer = init_optimizer(self.private_model, args)

        self.shared_model = get_model(args.shared_model, num_classes=args.num_classes)
        self.shared_optimizer = init_optimizer(self.shared_model, args)

        # initialize the weight vector to constant values of 0.5
        self.learned_weight_for_inference = torch.tensor([0.5], requires_grad=True, device=self.device)
        self.optimizer_learned_weight_for_inference = torch.optim.SGD([self.learned_weight_for_inference], lr=1e-3)

        self.train_loader = train_loader
        self.validation_loader = adaptability_loader
        self.eval_loader = eval_loader

    def refresh_optimizer(self):
        self.private_optimizer = init_optimizer(self.private_model, self.args)
        self.shared_optimizer = init_optimizer(self.shared_model, self.args)

    def learn_weight_for_inference(self):
        """
        Learning for Adaptability
        @return:
        """
        self.private_model = self.private_model.to(self.device)
        self.private_model.eval()
        self.shared_model = self.shared_model.to(self.device)
        self.shared_model.eval()

        for _ in range(10):
            for data, target in self.validation_loader:
                self.optimizer_learned_weight_for_inference.zero_grad()
                data, target = data.to(self.device), target.to(self.device)
                output_private = self.private_model(data).detach()
                output_shared = self.shared_model(data).detach()

                ensemble_output = self.learned_weight_for_inference * output_private + (1 - self.learned_weight_for_inference) * output_shared
                loss = CE_Loss(ensemble_output, target)
                loss.backward()
                self.optimizer_learned_weight_for_inference.step()
                torch.clip_(self.learned_weight_for_inference.data, 0.0, 1.0)

        self.staged_learned_weight_inference.append(self.learned_weight_for_inference.cpu().data.item())
        self.private_model = self.private_model.cpu()
        self.shared_model = self.shared_model.cpu()

        print('client {0} learned weight for inference: {1}'.format(self.idx, self.learned_weight_for_inference.data.item()))
