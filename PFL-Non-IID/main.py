#!/usr/bin/env python
import argparse
import copy
import os
import random
import time
import warnings

import numpy as np
import torch
import torchvision
from torch.nn.init import xavier_normal_, kaiming_normal_

from data_loader import MyDataLoader
from flcore.servers.serveramp import FedAMP
from flcore.servers.serverprox import FedProx
from flcore.servers.serverrep import FedRep
from flcore.trainmodel.cifarnet import *
from flcore.trainmodel.models import *
from utils.mem_utils import MemReporter
from utils.result_utils import average_data

warnings.simplefilter("ignore")


def init_weights(model, init_type):
    if init_type not in ['none', 'xavier', 'kaiming']:
        raise ValueError('init must in "none", "xavier" or "kaiming"')

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'xavier':
                xavier_normal_(m.weight.data, gain=1.0)
            elif init_type == 'kaiming':
                kaiming_normal_(m.weight.data, nonlinearity='relu')

    if init_type != 'none':
        model.apply(init_func)


def run(args):
    time_list = []
    reporter = MemReporter()
    set_seed(args.seed)
    dl = MyDataLoader(args)

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()
        if type(args.model) == type(''):
            model_str = args.model

        # Generate args.model
        if 'cnn2' == model_str:
            args.model = CNN2(num_classes=args.num_classes).to(args.device)
        elif 'cnn2_bn' == model_str:
            args.model = CNN2_BN(num_classes=args.num_classes).to(args.device)
        elif model_str == "resnet":
            args.model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes).to(args.device)
        init_weights(args.model, 'kaiming')
        # select algorithm
        if args.algorithm == 'FedAMP':
            server = FedAMP(args, dl, i)

        elif args.algorithm == "FedProx":
            server = FedProx(args, dl, i)

        elif args.algorithm == "FedRep":
            if i == 0:
                args.predictor = copy.deepcopy(args.model.fc)
                args.model.fc = nn.Identity()
                args.model = LocalModel(args.model, args.predictor)
            server = FedRep(args, dl, i)

        server.train()

        time_list.append(time.time() - start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    # Global average
    average_data(dataset=args.dataset,
                 algorithm=args.algorithm,
                 goal=args.goal,
                 times=args.times,
                 length=args.global_rounds / args.eval_gap + 1)

    print("All done!")

    reporter.report()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if seed == -1:
        torch.backends.cudnn.deterministic = False
    else:
        torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--seed', default=69, type=int)
    parser.add_argument('--iid', default='0', type=str)
    parser.add_argument('--download', default=False, type=bool)
    parser.add_argument('-go', "--goal", type=str, default="test",
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="mnist")
    parser.add_argument('--ratio_train_adaptability', default='[1.0, 0.0]', type=str)
    parser.add_argument('-m', "--model", type=str, default="cnn2_bn")
    parser.add_argument('-p', "--predictor", type=str, default="cnn")
    parser.add_argument('-lbs', "--batch", type=int, default=64)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=1e-2,
                        help="Local learning rate")
    parser.add_argument('--lr_decay', default=0.99, type=float)
    parser.add_argument('-gr', "--global_rounds", type=int, default=1000)
    parser.add_argument('-ls', "--local_steps", type=int, default=5)
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-nc', "--nodes", type=int, default=10,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-dp', "--privacy", type=bool, default=False,
                        help="differential privacy")
    parser.add_argument('-dps', "--dp_sigma", type=float, default=0.0)
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='models')
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Dropout rate for clients")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    # pFedMe / PerAvg / FedProx / FedAMP / FedPHP
    parser.add_argument('-bt', "--beta", type=float, default=0.0,
                        help="Average moving parameter for pFedMe, Second learning rate of Per-FedAvg, \
                        or L1 regularization weight of FedTransfer")
    parser.add_argument('-lam', "--lamda", type=float, default=1.0,
                        help="Regularization weight for pFedMe and FedAMP")
    parser.add_argument('-mu', "--mu", type=float, default=0,
                        help="Proximal rate for FedProx")
    parser.add_argument('-K', "--K", type=int, default=5,
                        help="Number of personalized training steps for pFedMe")
    parser.add_argument('-lrp', "--p_learning_rate", type=float, default=0.01,
                        help="personalized learning rate to caculate theta aproximately using K steps")
    # FedFomo
    parser.add_argument('-M', "--M", type=int, default=5,
                        help="Server only sends M client models to one client at each round")
    # FedMTL
    parser.add_argument('-itk', "--itk", type=int, default=4000,
                        help="The iterations for solving quadratic subproblems")
    # FedAMP
    parser.add_argument('-alk', "--alphaK", type=float, default=1.0,
                        help="lambda/sqrt(GLOABL-ITRATION) according to the paper")
    parser.add_argument('-sg', "--sigma", type=float, default=1.0)
    # APFL
    parser.add_argument('-al', "--alpha", type=float, default=1.0)
    # Ditto / FedRep
    parser.add_argument('-pls', "--plocal_steps", type=int, default=1)

    parser.add_argument('--log_root', default='logs', type=str, help='the root path of log directory')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not available.\n")
        args.device = "cpu"

    print("=" * 50)

    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch))
    print("Local steps: {}".format(args.local_steps))
    print("Local learning rate: {}".format(args.local_learning_rate))
    print("Total number of clients: {}".format(args.nodes))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Client drop rate: {}".format(args.client_drop_rate))
    print("Time select: {}".format(args.time_select))
    print("Time threshold: {}".format(args.time_threthold))
    print("Global rounds: {}".format(args.global_rounds))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Local model: {}".format(args.model))
    print("Using device: {}".format(args.device))

    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    if args.algorithm == "pFedMe":
        print("Average moving parameter beta: {}".format(args.beta))
        print("Regularization rate: {}".format(args.lamda))
        print("Number of personalized training steps: {}".format(args.K))
        print("personalized learning rate to calculate theta: {}".format(args.p_learning_rate))
    elif args.algorithm == "PerAvg":
        print("Second learning rate beta: {}".format(args.beta))
    elif args.algorithm == "FedProx":
        print("Proximal rate: {}".format(args.mu))
    elif args.algorithm == "FedFomo":
        print("Server sends {} models to one client at each round".format(args.M))
    elif args.algorithm == "FedMTL":
        print("The iterations for solving quadratic subproblems: {}".format(args.itk))
    elif args.algorithm == "FedAMP":
        print("alphaK: {}".format(args.alphaK))
        print("lambda: {}".format(args.lamda))
        print("sigma: {}".format(args.sigma))
    elif args.algorithm == "APFL":
        print("alpha: {}".format(args.alpha))
    elif args.algorithm == "Ditto":
        print("plocal_steps: {}".format(args.plocal_steps))
        print("mu: {}".format(args.mu))
    elif args.algorithm == "FedRep":
        print("plocal_steps: {}".format(args.plocal_steps))
    elif args.algorithm == "FedPHP":
        print("mu: {}".format(args.mu))
        print("lambda: {}".format(args.lamda))
    print("=" * 50)

    if args.dataset.lower() == 'cifar-100':
        args.num_classes = 100
    else:
        args.num_classes = 10

    run(args)

    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    # print(f"\nTotal time cost: {round(time.time()-total_start, 2)}s.")
