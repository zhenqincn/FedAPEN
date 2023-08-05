import os
import platform
import random

import numpy as np
import json
import torch.utils.data
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import transforms, datasets


def load_dataset(dst_name: str = 'cifar-10', dst_path=None, download=False):
    if dst_path is None:
        if platform.system().lower() == 'windows':
            import getpass
            dst_path = r'C:\Users\{}\.dataset'.format(getpass.getuser())
        else:
            import pwd
            user_name = pwd.getpwuid(os.getuid())[0]
            dst_path = r'/home/{}/.dataset'.format(user_name)
    if dst_name.lower() == 'cifar-10':
        transform_crop = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_no_crop = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_set_crop = datasets.CIFAR10(root=dst_path, train=True, transform=transform_crop, download=download)
        train_set_no_crop = datasets.CIFAR10(root=dst_path, train=True, transform=transform_no_crop, download=download)
        eval_set_crop = datasets.CIFAR10(root=dst_path, train=False, transform=transform_crop, download=download)
        eval_set_no_crop = datasets.CIFAR10(root=dst_path, train=False, transform=transform_no_crop, download=download)

    elif dst_name.lower() == 'mnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        transform_eval = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_set = datasets.MNIST(root=dst_path, train=True, transform=transform_train, download=download)
        eval_set = datasets.MNIST(root=dst_path, train=False, transform=transform_eval, download=download)
    elif dst_name.lower() == 'cifar-100':
        transform_crop = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_no_crop = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_set_crop = datasets.CIFAR100(root=dst_path, train=True, transform=transform_crop, download=download)
        train_set_no_crop = datasets.CIFAR100(root=dst_path, train=True, transform=transform_no_crop, download=download)
        eval_set_crop = datasets.CIFAR100(root=dst_path, train=False, transform=transform_crop, download=download)
        eval_set_no_crop = datasets.CIFAR100(root=dst_path, train=False, transform=transform_no_crop, download=download)
    else:
        raise ValueError('the dataset must be cifar-10, mnist or cifar-100')
    return train_set_crop, train_set_no_crop, eval_set_crop, eval_set_no_crop


def gen_len_splits(num_total, num_parts):
    quotient = num_total // num_parts
    remainder = num_total % num_parts
    len_splits = [quotient for _ in range(num_parts)]
    len_splits[0] += remainder
    return len_splits


def partition_idx_labelnoniid(y, n_parties, label_num, num_classes):
    K = num_classes
    if label_num == K:
        net_dataidx_map = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
        for i in range(10):
            idx_k = np.where(y == i)[0]
            np.random.shuffle(idx_k)
            split = np.array_split(idx_k, n_parties)
            for j in range(n_parties):
                net_dataidx_map[j] = np.append(net_dataidx_map[j], split[j])
    else:
        loop_cnt = 0
        while loop_cnt < 1000:
            times = [0 for _ in range(num_classes)]
            contain = []
            for i in range(n_parties):
                current = [i % K]
                times[i % K] += 1
                j = 1
                while j < label_num:
                    ind = random.randint(0, K - 1)
                    if ind not in current:
                        j = j + 1
                        current.append(ind)
                        times[ind] += 1
                contain.append(current)
            if len(np.where(np.array(times) == 0)[0]) == 0:
                break
            else:
                loop_cnt += 1

        # tackle down the issue that there is zero elements in array `times`
        zero_indices = np.where(np.array(times) == 0)[0]
        for zero_time_label in zero_indices:
            client_indices = np.array([idx for idx in range(n_parties)])
            np.random.shuffle(client_indices)
            for i in client_indices:
                selected_indices_time_over_one = np.where(np.array([times[label_idx] for label_idx in contain[i]]) > 1)[
                    0]
                if len(selected_indices_time_over_one) > 0:
                    j = selected_indices_time_over_one[0]
                    times[contain[i][j]] -= 1
                    contain[i].pop(j)
                    contain[i].append(zero_time_label)
                    times[zero_time_label] += 1
                    break

        net_dataidx_map = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
        for i in range(K):
            idx_k = np.where(y == i)[0]
            np.random.shuffle(idx_k)
            split = np.array_split(idx_k, times[i])
            ids = 0
            for j in range(n_parties):
                if i in contain[j]:
                    net_dataidx_map[j] = np.append(net_dataidx_map[j], split[ids])
                    ids += 1
    return net_dataidx_map


def partition_idx_labeldir(y, n_parties, alpha, num_classes):
    min_size = 0
    min_require_size = 10
    K = num_classes
    N = y.shape[0]
    net_dataidx_map = {}

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(n_parties)]
        for k in range(K):
            idx_k = np.where(y == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_parties))
            # logger.info("proportions1: ", proportions)
            # logger.info("sum pro1:", np.sum(proportions))
            # Balance
            proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
            # logger.info("proportions2: ", proportions)
            proportions = proportions / proportions.sum()
            # logger.info("proportions3: ", proportions)
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            # logger.info("proportions4: ", proportions)
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
            # if K == 2 and n_parties <= 10:
            #     if np.min(proportions) < 200:
            #         min_size = 0
            #         break
    for j in range(n_parties):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
    return net_dataidx_map


class MyDataLoader(object):
    """
    The data loader for split dataset, containing an adaptability set cut from the origin train set.
    The data samples from original train set and eval set are first collected, then distributed to each client.
    For each client, the data samples received by it will be divided into train set, adaptability set (if needed) and test set.
    """
    def __init__(self, args) -> None:
        train_set_crop, train_set_no_crop, eval_set_crop, eval_set_no_crop = load_dataset(args.dataset, None, download=args.download)
        set_ratio = json.loads(args.ratio_train_adaptability)
        train_ratio, adaptability_ratio = set_ratio[0], set_ratio[1]
        if args.iid == "0" or args.iid == 0:
            universal_set_crop = ConcatDataset([train_set_crop, eval_set_crop])
            universal_set_no_crop = ConcatDataset([train_set_no_crop, eval_set_no_crop])
            indices = np.array([i for i in range(len(universal_set_crop))])
            np.random.shuffle(indices)
            indices_list = np.array_split(indices, args.nodes)

            self.list_partitioned_set_train = []
            self.list_partitioned_set_adaptability = []
            self.list_partitioned_set_eval = []

            for client_id in range(args.nodes):
                num_train_cur_client = int(
                    len(indices_list[client_id]) * float(len(train_set_crop)) / len(universal_set_crop))
                num_adaptability_cur_client = int(num_train_cur_client * adaptability_ratio)
                num_train_cur_client -= num_adaptability_cur_client
                num_eval_cur_client = len(indices_list[client_id]) - num_train_cur_client - num_adaptability_cur_client

                if num_adaptability_cur_client > 0:
                    self.list_partitioned_set_train.append(
                        Subset(universal_set_crop, indices=indices_list[client_id][:num_train_cur_client]))
                    self.list_partitioned_set_adaptability.append(Subset(universal_set_no_crop, indices=indices_list[client_id][num_train_cur_client:num_train_cur_client + num_adaptability_cur_client]))
                    self.list_partitioned_set_eval.append(Subset(universal_set_no_crop, indices=indices_list[client_id][num_train_cur_client + num_adaptability_cur_client:]))
                else:
                    self.list_partitioned_set_train.append(
                        Subset(universal_set_crop, indices=indices_list[client_id][:num_train_cur_client]))
                    self.list_partitioned_set_adaptability.append([])
                    self.list_partitioned_set_eval.append(
                        Subset(universal_set_no_crop, indices=indices_list[client_id][num_train_cur_client:]))

        else:
            universal_set_crop = ConcatDataset([train_set_crop, eval_set_crop])
            universal_set_no_crop = ConcatDataset([train_set_no_crop, eval_set_no_crop])
            y_universal = np.array([item[1] for item in universal_set_crop])
            if 'dir' in args.iid:
                alpha = float(args.iid[3:])
                print('alpha', alpha)
                map_client_idx = partition_idx_labeldir(y_universal, args.nodes, alpha=alpha,
                                                        num_classes=args.num_classes)
            else:
                degree = int(args.iid)
                if args.num_classes == 100:  # for CIFAR-100
                    degree *= 10
                map_client_idx = partition_idx_labelnoniid(y_universal, args.nodes, degree, num_classes=args.num_classes)

            indices_list = []
            for _, v in map_client_idx.items():
                np.random.shuffle(v)
                indices_list.append(v)

            self.list_partitioned_set_train = []
            self.list_partitioned_set_adaptability = []
            self.list_partitioned_set_eval = []

            for client_id in range(args.nodes):
                num_train_cur_client = int(
                    len(indices_list[client_id]) * float(len(train_set_crop)) / len(universal_set_crop))
                num_adaptability_cur_client = int(num_train_cur_client * adaptability_ratio)
                num_train_cur_client -= num_adaptability_cur_client

                if num_adaptability_cur_client > 0:
                    self.list_partitioned_set_train.append(
                        Subset(universal_set_crop, indices=indices_list[client_id][:num_train_cur_client]))
                    self.list_partitioned_set_adaptability.append(Subset(universal_set_no_crop, indices=indices_list[client_id][num_train_cur_client:num_train_cur_client + num_adaptability_cur_client]))
                    self.list_partitioned_set_eval.append(Subset(universal_set_no_crop, indices=indices_list[client_id][num_train_cur_client + num_adaptability_cur_client:]))
                else:
                    self.list_partitioned_set_train.append(
                        Subset(universal_set_crop, indices=indices_list[client_id][:num_train_cur_client]))
                    self.list_partitioned_set_adaptability.append([])
                    self.list_partitioned_set_eval.append(
                        Subset(universal_set_no_crop, indices=indices_list[client_id][num_train_cur_client:]))

        self.train_loader_list = [DataLoader(train_subset, args.batch, shuffle=True, pin_memory=True) for train_subset
                                  in self.list_partitioned_set_train]
        if adaptability_ratio > 0:
            self.adaptability_loader_list = [
                DataLoader(adaptability_subset, min(args.batch, len(adaptability_subset)), shuffle=True, pin_memory=True)
                for adaptability_subset
                in self.list_partitioned_set_adaptability]
        else:
            self.adaptability_loader_list = [[] for _ in range(args.nodes)]
        self.eval_loader_list = [DataLoader(eval_subset, args.batch, shuffle=True, pin_memory=True) for eval_subset in
                                 self.list_partitioned_set_eval]

        self.train_loader_complete = DataLoader(train_set_crop, args.batch, shuffle=True, pin_memory=True)
        self.eval_loader_complete = DataLoader(eval_set_no_crop, args.batch, shuffle=True, pin_memory=True)
