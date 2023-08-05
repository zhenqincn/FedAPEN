import codecs
import copy
import json
import os
import time

import h5py
import numpy as np
import torch

from flcore.trainmodel.models import LocalModel


class Server(object):
    def __init__(self, args, my_data_loader, times):
        # Set up the main attributes
        self.device = args.device
        self.dataset = args.dataset
        self.my_data_loader = my_data_loader
        self.global_rounds = args.global_rounds
        self.local_steps = args.local_steps
        self.batch = args.batch
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.nodes = args.nodes
        self.join_ratio = args.join_ratio
        self.join_clients = int(self.nodes * self.join_ratio)
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.algorithm = args.algorithm
        self.log_root = args.log_root
        self.val_acc = [[] for _ in range(args.nodes)]

        self.time_stamp = int(time.time())
        if not os.path.exists(os.path.join(args.log_root, 'models')):
            os.makedirs(os.path.join(args.log_root, str(self.time_stamp), 'models'))
        with open(os.path.join(args.log_root, str(self.time_stamp), 'params.log'), 'w') as writer:
            for k, v in args.__dict__.items():
                if k == 'model':
                    if isinstance(v, LocalModel):
                        print(k, ':', v.model_name, file=writer)
                    else:
                        print(k, ':', str(v.__class__.__name__).lower(), file=writer)
                else:
                    print(k, ':', v, file=writer)

    def set_clients(self, args, clientObj):
        for i, train_slow, send_slow in zip(range(self.nodes), self.train_slow_clients, self.send_slow_clients):
            client = clientObj(args,
                               id=i,
                               train_loader=self.my_data_loader.train_loader_list[i],
                               test_loader=self.my_data_loader.eval_loader_list[i],
                               train_slow=train_slow,
                               send_slow=send_slow)
            self.clients.append(client)

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.nodes)]
        idx = [i for i in range(self.nodes)]
        idx_ = np.random.choice(idx, int(slow_rate * self.nodes))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        if self.join_clients < len(self.clients):
            selected_clients = list(np.random.choice(self.clients, self.join_clients, replace=False))
            return selected_clients
        else:
            return self.clients

    def send_models(self):
        assert (len(self.selected_clients) > 0)

        for client in self.selected_clients:
            client.set_parameters(self.global_model)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_weights = []
        tot_samples = 0
        self.uploaded_ids = []
        self.uploaded_models = []
        for client in self.selected_clients:
            self.uploaded_weights.append(len(client.train_loader.dataset))
            tot_samples += len(client.train_loader.dataset)
            self.uploaded_ids.append(client.id)
            self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            # average all the weights
            self.uploaded_weights[i] = 1.0 / len(self.selected_clients)

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        weights_avg = copy.deepcopy(self.uploaded_models[0].state_dict())
        for key in weights_avg.keys():
            for i in range(1, len(self.uploaded_models)):
                weights_avg[key] += self.uploaded_models[i].state_dict()[key].detach()
            weights_avg[key] = torch.div(weights_avg[key], len(self.uploaded_models))
        self.global_model.load_state_dict(weights_avg)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)

    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

    def test_metrics(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.selected_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)

        ids = [c.id for c in self.selected_clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_accuracy_and_loss(self):
        num_samples = []
        losses = []
        for c in self.selected_clients:
            cl, ns = c.train_accuracy_and_loss()
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.selected_clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate(self):
        stats = self.test_metrics()
        acc_list = np.array(stats[2]) / np.array(stats[1])
        test_acc = np.average(acc_list)

        test_auc = sum(stats[3]) * 1.0 / sum(stats[1])

        self.rs_test_acc.append(test_acc)
        self.rs_test_auc.append(test_auc)

        print("Average Test Accuracy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))

        for client_idx, val_acc in enumerate(acc_list):
            self.val_acc[client_idx].append(val_acc)
        with codecs.open(os.path.join(self.log_root, str(self.time_stamp), 'summary.json'), 'w') as writer:
            dic = {
                "val_acc": self.val_acc,
            }
            json.dump(dic, writer)

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accuracy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))
