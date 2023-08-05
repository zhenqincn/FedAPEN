import copy
import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
from sklearn import metrics


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_loader, test_loader, **kwargs):
        self.args = args
        self.model = copy.deepcopy(args.model)
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_loader = train_loader
        self.train_samples = len(train_loader.dataset)
        self.test_loader = test_loader
        self.test_samples = len(test_loader.dataset)
        self.batch = args.batch
        self.learning_rate = args.local_learning_rate
        self.local_steps = args.local_steps

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.privacy = args.privacy
        self.dp_sigma = args.dp_sigma

    def load_train_data(self, batch=None):
        # if batch == None:
        #     batch = self.batch
        # train_data = read_client_data(self.dataset, self.id, is_train=True)
        # return DataLoader(train_data, batch, drop_last=True, shuffle=True)
        return self.train_loader

    def load_test_data(self, batch=None):
        # if batch == None:
        #     batch = self.batch
        # test_data = read_client_data(self.dataset, self.id, is_train=False)
        # return DataLoader(test_data, batch, drop_last=True, shuffle=True)
        return self.test_loader

    def set_parameters(self, model):
        self.model.load_state_dict(model.state_dict())
        # for new_param, old_param in zip(model.parameters(), self.model.parameters()):
        #     old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        # 这里的full指的是全部划分给当前client的数据
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloaderfull:
                x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(F.softmax(output).detach().cpu().numpy())
                y_true.append(label_binarize(y.detach().cpu().numpy(), classes=np.arange(self.num_classes)))

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        return test_acc, test_num, auc

    def save_item(self, item, item_name):
        item_path = os.path.join(self.save_folder_name, self.dataset)
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_model(self, model_name):
        model_path = os.path.join("models", self.dataset)
        return torch.load(os.path.join(model_path, "client_" + str(self.id) + "_" + model_name + ".pt"))
