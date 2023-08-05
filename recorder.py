import torch
import os
import time
import codecs
import json
import numpy as np
from node import Node
import torch.nn as nn


KL_Loss = nn.KLDivLoss(reduction='batchmean')
Softmax = nn.Softmax(dim=1)
LogSoftmax = nn.LogSoftmax(dim=1)
CE_Loss = nn.CrossEntropyLoss()


class Recorder(object):
    def __init__(self, args):
        self.args = args
        self.time_stamp = int(time.time())
        
        self.train_record = [[] for _ in range(args.nodes)]
        self.val_loss = [[] for _ in range(args.nodes)]
        self.val_acc = [[] for _ in range(args.nodes)]
        self.acc_best = torch.zeros(self.args.nodes)
        self.avg_acc_best = 0

        # only used for mutual learning
        self.val_acc_mutual_private = [[] for _ in range(args.nodes)]
        self.val_acc_mutual_shared = [[] for _ in range(args.nodes)]
        self.val_acc_mutual_ensemble = [[] for _ in range(args.nodes)]
        self.val_acc_mutual_ensemble_adaptive = [[] for _ in range(args.nodes)]
        
        self.val_loss_mutual_private = [[] for _ in range(args.nodes)]
        self.val_loss_mutual_shared = [[] for _ in range(args.nodes)]
        
        self.get_a_better = [False for _ in range(self.args.nodes)]

        self.server_node = None
        self.node_list = None

        if not os.path.exists(os.path.join(args.log_root, 'models')):
            os.makedirs(os.path.join(args.log_root, str(self.time_stamp), 'models'))
        
        with open(os.path.join(args.log_root, str(self.time_stamp), 'params.log'), 'w') as writer:
            for k, v in args.__dict__.items():
                print(k, ':', v, file=writer)
    
    def _eval_single_model(self, node):
        if self.args.algorithm.lower() in ['individual', 'apfl']:
            node.private_model = node.private_model.to(node.device)
            node.private_model.eval()
            model_name = node.args.private_model
            model = node.private_model
        else: 
            node.shared_model = node.shared_model.to(node.device)
            node.shared_model.eval()
            model_name = node.args.shared_model
            model = node.shared_model
        total_loss = 0.0
        correct = 0.0
        with torch.no_grad():
            for idx, (data, target) in enumerate(node.eval_loader):
                data, target = data.to(node.device), target.to(node.device)
                output = model(data)
                total_loss += CE_Loss(output, target)
                pred = output.argmax(dim=1)
                correct += pred.eq(target.view_as(pred)).sum().item()
            total_loss = total_loss / (idx + 1)
            acc = correct / len(node.eval_loader.dataset) * 100
        self.val_loss[node.idx].append(total_loss.item())
        self.val_acc[node.idx].append(acc)

        # detach the model from GPU device
        if self.args.algorithm.lower() in ['individual', 'apfl']:
            node.private_model = node.private_model.cpu()
        else:  # server node
            node.shared_model = node.shared_model.cpu()
            
        # save model
        if self.args.model_save == 'best':
            if self.val_acc[node.idx][-1] > self.acc_best[node.idx]:
                self.get_a_better[node.idx] = True
                self.acc_best[node.idx] = self.val_acc[node.idx][-1]
                torch.save(model.state_dict(), os.path.join(self.args.log_root, str(self.time_stamp), 'models', 'Node{:d}_{:s}.pt'.format(node.idx, model_name)))
                print('-----------------------')
                print('Node{:d}: A Better Accuracy: {:.3f}%! Model Saved!'.format(node.idx, self.acc_best[node.idx]))
                print('-----------------------')
            else:
                print('Node{:d}: Accuracy: {:.3f}%'.format(node.idx, self.val_acc[node.idx][-1]))

    def _eval_ensemble_model(self, node: Node):
        node.private_model = node.private_model.to(node.device)
        node.private_model.eval()
        node.shared_model = node.shared_model.to(node.device)
        node.shared_model.eval()
        
        if node.args.algorithm in ['learned_adaptive_training', 'learned_adaptive', 'equal_training']:
            weight_private = node.learned_weight_for_inference.cpu().item()
        else:
            raise AttributeError()
        print('Node {0}, eval private weight: {1}'.format(node.idx, weight_private))
        
        total_loss_private = 0.0
        total_loss_shared = 0.0
        correct_private = 0.0
        correct_shared = 0.0
        correct_ensemble = 0.0
        correct_ensemble_adaptive = 0.0
        with torch.no_grad():
            for idx, (data, target) in enumerate(node.eval_loader):
                data, target = data.to(node.device), target.to(node.device)
                output_private = node.private_model(data)
                output_shared = node.shared_model(data)
                
                ce_private = CE_Loss(output_private, target)
                kl_private = KL_Loss(LogSoftmax(output_private), Softmax(output_shared.detach()))
                ce_shared = CE_Loss(output_shared, target)
                kl_shared = KL_Loss(LogSoftmax(output_shared), Softmax(output_private.detach()))
                total_loss_private += ce_private + kl_private
                total_loss_shared += ce_shared + kl_shared
                
                pred_private = output_private.argmax(dim=1)
                pred_shared = output_shared.argmax(dim=1)
                pred_ensemble = (output_private + output_shared).argmax(dim=1)
                # adaptive 
                pred_ensemble_adaptive = (weight_private * output_private + (1 - weight_private) * output_shared).argmax(dim=1)
                
                correct_private += pred_private.eq(target.view_as(pred_private)).sum().item()
                correct_shared += pred_shared.eq(target.view_as(pred_shared)).sum().item()
                correct_ensemble += pred_ensemble.eq(target.view_as(pred_ensemble)).sum().item()
                correct_ensemble_adaptive += pred_ensemble_adaptive.eq(target.view_as(pred_ensemble_adaptive)).sum().item()
            total_loss_private = total_loss_private / (idx + 1)
            total_loss_shared = total_loss_shared / (idx + 1)
            
            acc_private = correct_private / len(node.eval_loader.dataset) * 100
            acc_shared = correct_shared / len(node.eval_loader.dataset) * 100
            acc_ensemble = correct_ensemble / len(node.eval_loader.dataset) * 100
            acc_ensemble_adaptive = correct_ensemble_adaptive / len(node.eval_loader.dataset) * 100
            
        self.val_loss_mutual_private[node.idx].append(total_loss_private.item())
        self.val_loss_mutual_shared[node.idx].append(total_loss_shared.item())
        
        self.val_acc_mutual_private[node.idx].append(acc_private)
        self.val_acc_mutual_shared[node.idx].append(acc_shared)
        self.val_acc_mutual_ensemble[node.idx].append(acc_ensemble)
        self.val_acc_mutual_ensemble_adaptive[node.idx].append(acc_ensemble_adaptive)

        # detach the model from GPU device
        node.private_model = node.private_model.cpu()
        node.shared_model = node.shared_model.cpu()

    def eval(self, node):
        if self.args.algorithm in ['learned_adaptive_training', 'equal_training', 'learned_adaptive', 'mutual', 'heuristic_mutual']:
            self._eval_ensemble_model(node)
        else:
            self._eval_single_model(node)

    def add_train_record(self, record: dict):
        idx = record['idx']
        self.train_record[idx].append(record)

    def register_server_node(self, node):
        self.server_node = node

    def register_node_list(self, node_list):
        self.node_list = node_list

    def summary(self, cur_round):
        if self.args.algorithm in ['learned_adaptive_training', 'equal_training', 'learned_adaptive']:
            learned_weight_inference = [node.staged_learned_weight_inference for node in self.node_list]
            print('The {:d}-th round, Average Private ACC: {:.3f}%, Average shared ACC: {:.3f}%, Average Ensemble ACC: {:.3f}%, Average Adaptive Ensemble ACC: {:.3f}%'.format(
                    cur_round + 1,
                    np.mean([self.val_acc_mutual_private[idx][-1] for idx in range(self.args.nodes)]),
                    np.mean([self.val_acc_mutual_shared[idx][-1] for idx in range(self.args.nodes)]),
                    np.mean([self.val_acc_mutual_ensemble[idx][-1] for idx in range(self.args.nodes)]),
                    np.mean([self.val_acc_mutual_ensemble_adaptive[idx][-1] for idx in range(self.args.nodes)])
                ))
            if (cur_round + 1) % 10 == 0:
                with codecs.open(os.path.join(self.args.log_root, str(self.time_stamp), 'summary.json'), 'w') as writer:
                    dic = {
                        "train_record": self.train_record,
                        "val_acc_mutual_private": self.val_acc_mutual_private,
                        "val_acc_mutual_shared": self.val_acc_mutual_shared,
                        "val_acc_mutual_ensemble": self.val_acc_mutual_ensemble,
                        "val_acc_mutual_ensemble_adaptive": self.val_acc_mutual_ensemble_adaptive,
                        "learned_weight_inference": learned_weight_inference,
                    }
                    json.dump(dic, writer)
        else:
            print('The {:d}-th round, Average ACC: {:.3f}%!'.format(cur_round + 1, np.mean([self.val_acc[idx][-1] for idx in range(self.args.nodes)])))
            if self.args.algorithm == 'apfl':
                learned_weight_inference = [node.staged_learned_weight_inference for node in self.node_list]
            else:
                learned_weight_inference = []
            if (cur_round + 1) % 1 == 0:
                with codecs.open(os.path.join(self.args.log_root, str(self.time_stamp), 'summary.json'), 'w') as writer:
                    dic = {
                        "train_record": self.train_record,
                        "val_acc": self.val_acc,
                        "val_loss": self.val_loss,
                        "learned_weight_inference": learned_weight_inference
                    }
                    json.dump(dic, writer)

        
    