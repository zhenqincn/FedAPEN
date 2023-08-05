import matplotlib.pyplot as plt
import os
import codecs
import json
import numpy as np

last_iid = None
last_dataset = None
optional_color = ['#ff0080', '#1a6fdf', '#37ad6b', '#F57C00', '#673AB7']
line_cnt = 0


def plot_learned_weight_inference(root_dir, time_stamp):
    with codecs.open(os.path.join(root_dir, time_stamp, 'params.log'), 'r') as reader:
        for line in reader.readlines():
            if 'iid' in line:
                line = line.strip()
                colon_index = line.index(':')
                cur_iid = line[colon_index + 2:]
                break
    with codecs.open(os.path.join(root_dir, time_stamp, 'summary.json'), 'r') as reader:
        summary = json.load(reader)
    weight_matrix = summary['learned_weight_inference']
    x = [i for i in range(1, len(weight_matrix[0]) + 1)]
    if len(np.where(np.array(weight_matrix) < 0)[0]) > 0:
        raise ValueError('there exists weight smaller than 0')
    if len(np.where(np.array(weight_matrix) > 1)[0]) > 0:
        raise ValueError('there exists weight larger than 1')
    if 'dir' in cur_iid:
        label = '$\\alpha$ = {0}'.format(cur_iid.replace('dir', ''))
    else:
        if '0' == cur_iid:
            label = '$c$ = 1.0'
        else:
            label = '$c$ = {:.1f}'.format(float(cur_iid) / 10)
    global line_cnt
    plt.plot(x, np.array(weight_matrix).mean(axis=0), label=label, c=optional_color[line_cnt], lw=2.5)
    line_cnt += 1


def plot_model_acc(root_dir, log_dir, verbose=True, is_plot=True, rounds=100, return_all=False):
    """
        @param return_all: if true, return mean, max and min of all approaches, else, return max(mean), max and min of one approach
    """
    with codecs.open(os.path.join(root_dir, log_dir, 'params.log'), 'r') as reader:
        for line in reader.readlines():
            if 'algorithm' in line:
                line = line.strip()
                colon_index = line.index(':')
                algorithm = line[colon_index + 2:]
            # check the consistence of non-iid degree
            elif line.startswith('iid'):
                line = line.strip()
                colon_index = line.index(':')
                cur_iid = line[colon_index + 2:]
                global last_iid
                if last_iid is None:
                    last_iid = cur_iid
                elif last_iid != cur_iid:
                    raise ValueError('a result from different iid is staged:', log_dir)
            elif 'dataset' in line and 'log_root' not in line:  # prevent the situation that `log_root` contains the word `dataset`
                line = line.strip()
                colon_index = line.index(':')
                cur_dataset = line[colon_index + 2:]
                global last_dataset
                if last_dataset is None:
                    last_dataset = cur_dataset
                elif last_dataset != cur_dataset:
                    raise ValueError('a result from different dataset is staged:', log_dir)
    with codecs.open(os.path.join(root_dir, log_dir, 'summary.json'), 'r') as reader:
        summary = json.load(reader)
        if algorithm in ['learned_adaptive_training', 'equal_training', 'learned_adaptive', 'mutual', 'heuristic_mutual']:
            val_acc_mutual_private = np.array(summary['val_acc_mutual_private'])
            mean_acc_private = np.mean(val_acc_mutual_private, axis=0)
            if len(mean_acc_private) != rounds:
                raise ValueError()
            std_acc_private = np.std(val_acc_mutual_private[:, np.argmax(mean_acc_private)])

            val_acc_mutual_shared = np.array(summary['val_acc_mutual_shared'])
            mean_acc_shared = np.mean(val_acc_mutual_shared, axis=0)
            if len(mean_acc_shared) != rounds:
                raise ValueError()
            std_acc_shared = np.std(val_acc_mutual_shared[:, np.argmax(mean_acc_shared)])

            val_acc_mutual_ensemble = np.array(summary['val_acc_mutual_ensemble'])
            mean_acc_ensemble = np.mean(val_acc_mutual_ensemble, axis=0)
            if len(mean_acc_ensemble) != rounds:
                raise ValueError()
            std_acc_ensemble = np.std(val_acc_mutual_ensemble[:, np.argmax(mean_acc_ensemble)])

            val_acc_mutual_ensemble_adaptive = np.array(summary['val_acc_mutual_ensemble_adaptive'])
            mean_acc_ensemble_adaptive = np.mean(val_acc_mutual_ensemble_adaptive, axis=0)
            if len(mean_acc_ensemble_adaptive) != rounds:
                raise ValueError()
            std_acc_ensemble_adaptive = np.std(val_acc_mutual_ensemble_adaptive[:, np.argmax(mean_acc_ensemble_adaptive)])
            # print(std_acc_ensemble_adaptive)

            x = range(1, len(mean_acc_private) + 1)

            if algorithm == 'learned_adaptive':
                label_private = 'FML(Pri)'
                label_shared = 'FML(Sha)'
                label_ensemble = 'FML-EE'
                label_ensemble_adaptive = 'FML-AE'

            elif algorithm == 'learned_adaptive_training':
                label_private = 'FedAPEN(Pri)'
                label_shared = 'FedAPEN(Sha)'
                label_ensemble = 'FedAPEN-EE'
                label_ensemble_adaptive = 'FedAPEN'

            elif algorithm == 'equal_training':
                label_private = 'FedEN-ET (Pri)'
                label_shared = 'FedEN-ET (Sha)'
                label_ensemble = 'FedEN'
                label_ensemble_adaptive = 'FedEN (Adaptive Inf)'

            if verbose and is_plot:
                plt.plot(x, mean_acc_private, label=label_private)
                print('{:}: {:.2f} ± {:.2f}'.format(label_private, np.max(mean_acc_private), std_acc_private))
                plt.plot(x, mean_acc_shared, label=label_shared)
                print('{:}: {:.2f} ± {:.2f}'.format(label_shared, np.max(mean_acc_shared), std_acc_shared))
            if is_plot:
                plt.plot(x, mean_acc_ensemble, label=label_ensemble)
                print('{:}: {:.2f} ± {:.2f}'.format(label_ensemble, np.max(mean_acc_ensemble), std_acc_ensemble))
            if is_plot:
                for a, b in zip(x, mean_acc_ensemble_adaptive):
                    print(a, b)
                plt.plot(x, mean_acc_ensemble_adaptive, label=label_ensemble_adaptive)
                print('{:}: {:.2f} ± {:.2f}'.format(label_ensemble_adaptive, np.max(mean_acc_ensemble_adaptive), std_acc_ensemble_adaptive))
                print()
            max_index_private = np.argmax(mean_acc_private)
            max_index_shared = np.argmax(mean_acc_shared)
            max_index_ensemble = np.argmax(mean_acc_ensemble)
            max_index_ensemble_adaptive = np.argmax(mean_acc_ensemble_adaptive)
            
            if not return_all:
                return {
                    label_private: [np.max(mean_acc_private), np.max(val_acc_mutual_private[:, max_index_private]), np.min(val_acc_mutual_private[:, max_index_private])],
                    label_shared: [np.max(mean_acc_shared), np.max(val_acc_mutual_shared[:, max_index_shared]), np.min(val_acc_mutual_shared[:, max_index_shared])],
                    label_ensemble: [np.max(mean_acc_ensemble), np.max(val_acc_mutual_ensemble[:, max_index_ensemble]), np.min(val_acc_mutual_ensemble[:, max_index_ensemble])],
                    label_ensemble_adaptive: [np.max(mean_acc_ensemble_adaptive), np.max(val_acc_mutual_ensemble_adaptive[:, max_index_ensemble_adaptive]), np.min(val_acc_mutual_ensemble[:, max_index_ensemble_adaptive])]
                }
            else:
                if algorithm == 'learned_adaptive_training':
                    return label_ensemble_adaptive, [mean_acc_ensemble_adaptive, np.max(val_acc_mutual_ensemble_adaptive, axis=0), np.min(val_acc_mutual_ensemble_adaptive, axis=0)]
                elif algorithm == 'equal_training':
                    return label_ensemble, [mean_acc_ensemble, np.max(val_acc_mutual_ensemble, axis=0), np.min(val_acc_mutual_ensemble, axis=0)]
                else:
                    return label_private, [mean_acc_private, np.max(val_acc_mutual_private, axis=0), np.min(val_acc_mutual_private, axis=0)]
        else:
            val_acc = np.array(summary['val_acc'])
            mean_acc = np.mean(val_acc, axis=0)
            if len(mean_acc) != rounds:
                raise ValueError()

            idx_max_mean = np.argmax(mean_acc)
            std_acc = np.std(val_acc[:, idx_max_mean])
            # print(std_acc * 100)
            x = range(1, len(mean_acc) + 1)
            if is_plot:
                plt.plot(x, mean_acc, label=algorithm)
                print('{:}: {:.2f} ± {:.2f}'.format(algorithm, np.max(mean_acc), std_acc))
                print()
            if not return_all:
                return {
                    algorithm: [np.max(mean_acc), np.max(val_acc, axis=0), np.min(val_acc, axis=0)]
                }
            else:
                return algorithm, [mean_acc, np.max(val_acc, axis=0), np.min(val_acc, axis=0)]
