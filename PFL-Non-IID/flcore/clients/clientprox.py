import copy
import math
import time

import torch
import torch.nn as nn

from flcore.clients.clientbase import Client


class ClientProx(Client):
    def __init__(self, args, id, train_loader, test_loader, **kwargs):
        super().__init__(args, id, train_loader, test_loader, **kwargs)

        self.mu = args.mu

        self.global_params = copy.deepcopy(list(self.model.parameters()))

        self.loss = nn.CrossEntropyLoss()

    def train(self, epoch):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate * math.pow(self.args.lr_decay, epoch))
        trainloader = self.load_train_data()
        start_time = time.time()

        self.model.train()

        for step in range(self.local_steps):
            for x, y in trainloader:
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x)

                proximal_term = 0.0
                for w, w_t in zip(self.model.parameters(), self.global_params):
                    proximal_term += (w - w_t).norm(2)
                    # proximal_term += torch.sum(torch.square((w - w_t)))

                loss = self.loss(output, y) + (self.mu / 2) * proximal_term
                loss.backward()
                self.optimizer.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_parameters(self, model):
        for new_param, global_param, param in zip(model.parameters(), self.global_params, self.model.parameters()):
            global_param.data = new_param.data.clone()
            param.data = new_param.data.clone()
