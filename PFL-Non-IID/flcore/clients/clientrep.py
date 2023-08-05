import math
import time

import numpy as np
import torch
import torch.nn as nn

from flcore.clients.clientbase import Client


class ClientRep(Client):
    def __init__(self, args, id, train_loader, test_loader, **kwargs):
        super().__init__(args, id, train_loader, test_loader, **kwargs)

        self.loss = nn.CrossEntropyLoss()

        self.plocal_steps = args.plocal_steps

    def train(self, epoch):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate * math.pow(self.args.lr_decay, epoch))
        trainloader = self.load_train_data()
        start_time = time.time()
        # self.model.to(self.device)
        self.model.train()

        for param in self.model.base.parameters():
            param.requires_grad = False
        for param in self.model.predictor.parameters():
            param.requires_grad = True

        for step in range(self.plocal_steps):  # update predictor (head)
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()

        max_local_steps = self.local_steps
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for param in self.model.base.parameters():
            param.requires_grad = True
        for param in self.model.predictor.parameters():
            param.requires_grad = False

        for step in range(max_local_steps):  # update feature extractor (representation)
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_parameters(self, base):
        for new_param, old_param in zip(base.parameters(), self.model.base.parameters()):
            old_param.data = new_param.data.clone()
