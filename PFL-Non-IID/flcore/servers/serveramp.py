import copy
import math
import time

import numpy as np
import torch

from flcore.clients.clientamp import ClientAMP, weight_flatten
from flcore.servers.serverbase import Server


class FedAMP(Server):
    def __init__(self, args, my_data_loader, times):
        super().__init__(args, my_data_loader, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(args, ClientAMP)

        self.alphaK = args.alphaK
        self.sigma = args.sigma

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.nodes}")
        print("Finished creating server and clients.")

    def train(self):
        for i in range(self.global_rounds + 1):
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0 and i != 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train(epoch=i)

            self.receive_models()

        print("\nBest global accuracy.")
        print(max(self.rs_test_acc))

        self.save_results()
        self.save_global_model()

    def send_models(self):
        assert (len(self.selected_clients) > 0)

        if len(self.uploaded_models) > 0:
            for c in self.selected_clients:
                mu = copy.deepcopy(self.global_model)
                for param in mu.parameters():
                    param.data.zero_()

                coef = torch.zeros(self.join_clients)
                for j, mw in enumerate(self.uploaded_models):
                    if c.id != self.uploaded_ids[j]:
                        weights_i = weight_flatten(c.model)
                        weights_j = weight_flatten(mw)
                        sub = (weights_i - weights_j).view(-1)
                        sub = torch.dot(sub, sub)
                        coef[j] = self.alphaK * self.e(sub)
                    else:
                        coef[j] = 0
                coef_self = 1 - torch.sum(coef)

                for j, mw in enumerate(self.uploaded_models):
                    for param, param_j in zip(mu.parameters(), mw.parameters()):
                        param.data += coef[j] * param_j

                start_time = time.time()

                if c.send_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                c.set_parameters(mu, coef_self)

                c.send_time_cost['num_rounds'] += 1
                c.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def e(self, x):
        return math.exp(-x / self.sigma) / self.sigma
