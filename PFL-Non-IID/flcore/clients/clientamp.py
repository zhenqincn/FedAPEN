import copy
import math
import time

import torch
import torch.nn as nn
from flcore.clients.clientbase import Client


class ClientAMP(Client):
    def __init__(self, args, id, train_loader, test_loader, **kwargs):
        super().__init__(args, id, train_loader, test_loader, **kwargs)

        self.alphaK = args.alphaK
        self.lamda = args.lamda
        self.client_u = copy.deepcopy(self.model)

        self.loss = nn.CrossEntropyLoss()

    def train(self, epoch):
        optimizer = torch.optim.SGD(self.model.parameters(),
                                    lr=self.learning_rate * math.pow(self.args.lr_decay, epoch))
        trainloader = self.load_train_data()
        start_time = time.time()

        self.model.train()
        for _ in range(self.local_steps):
            for x, y in trainloader:
                x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)

                params = weight_flatten(self.model)
                params_ = weight_flatten(self.client_u)
                sub = params - params_
                loss += self.lamda / self.alphaK / 2 * torch.dot(sub, sub)

                loss.backward()
                optimizer.step()

        # self.model.cpu()
        del trainloader

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_parameters(self, model, coef_self):
        for new_param, old_param in zip(model.parameters(), self.client_u.parameters()):
            old_param.data = (new_param.data + coef_self * old_param.data).clone()


def weight_flatten(model):
    params = []
    for u in model.parameters():
        params.append(u.view(-1))
    params = torch.cat(params)

    return params
