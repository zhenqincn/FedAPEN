from tqdm import tqdm
import torch
import torch.nn as nn
from node import Node
import os
import numpy as np
from copy import deepcopy


KL_Loss = nn.KLDivLoss(reduction='batchmean')
Softmax = nn.Softmax(dim=1)
LogSoftmax = nn.LogSoftmax(dim=1)
CE_Loss = nn.CrossEntropyLoss()


def update_apfl_alpha(private_model, shared_model, alpha, eta):
    grad_alpha = 0
    for private_params, shared_params in zip(private_model.parameters(), shared_model.parameters()):
        dif = private_params.data - shared_params.data
        grad = alpha * private_params.grad.data + (1 - alpha) * shared_params.grad.data
        grad_alpha += dif.view(-1).T.dot(grad.view(-1))
    
    grad_alpha += 0.02 * alpha
    alpha_n = alpha - eta * grad_alpha
    alpha_n = np.clip(alpha_n.item(), 0.0, 1.0)
    return alpha_n


def apfl(node:Node, recorder, cur_round, args):
    node.private_model = node.private_model.to(node.device)
    node.private_model.train()
    node.shared_model = node.shared_model.to(node.device)
    node.shared_model.train()
    train_loader = node.train_loader
    description = 'Node{:d}: Local Epoch {:d}, loss_shared={:.4f} acc_shared={:.2f}, loss_private={:.4f} acc_private={:.2f}%'
    for epoch in range(node.args.local_epoch):
        total_loss_shared = 0.0
        correct_shared = 0.0
        num_trained = 0
        
        total_loss_private = 0.0
        correct_private = 0.0
        num_trained = 0
        
        with tqdm(train_loader) as bar_epoch:
            for idx, (data, target) in enumerate(bar_epoch):
                node.shared_optimizer.zero_grad()
                data, target = data.to(node.device), target.to(node.device)
                output_shared = node.shared_model(data)
                ce_shared = CE_Loss(output_shared, target)
                ce_shared.backward()
                node.shared_optimizer.step()  
                total_loss_shared += ce_shared
                pred_shared = output_shared.argmax(dim=1)
                correct_shared += pred_shared.eq(target.view_as(pred_shared)).sum()

                node.private_optimizer.zero_grad()
                data, target = data.to(node.device), target.to(node.device)
                output_private = node.private_model(data)
                ce_private = CE_Loss(output_private, target)
                ce_private.backward()
                node.private_optimizer.step()  
                total_loss_private += ce_private
                pred_private = output_private.argmax(dim=1)
                correct_private += pred_private.eq(target.view_as(pred_private)).sum()
                num_trained += len(data)
                
                if epoch == 0 and idx == 0:
                    node.alpha_apfl = update_apfl_alpha(node.private_model, node.shared_model, node.alpha_apfl, 0.01)
                    node.staged_learned_weight_inference.append(node.alpha_apfl)
                    print("client {0}, alpha: {1}".format(node.idx, node.alpha_apfl))
                
                bar_epoch.set_description(description.format(node.idx, epoch + 1, total_loss_shared / (idx + 1), correct_shared / num_trained * 100, total_loss_private / (idx + 1), correct_private / num_trained * 100))
    node.shared_model = node.shared_model.cpu()
    node.private_model = node.private_model.cpu()
    
    weights_avg = deepcopy(node.shared_model.state_dict())
    weights_pri = deepcopy(node.private_model.state_dict())
    for key in weights_avg.keys():
        weights_avg[key] = node.alpha_apfl * weights_pri[key].detach() +  (1 - node.alpha_apfl) * weights_avg[key].detach()
    node.private_model.load_state_dict(weights_avg)
    
    return {"idx": node.idx,
            "loss_shared": (total_loss_shared / (idx + 1)).detach().cpu().item(),
            "acc_shared": (correct_shared / num_trained * 100).detach().cpu().item(),
            "loss_private": (total_loss_private / (idx + 1)).detach().cpu().item(),
            "acc_private": (correct_private / num_trained * 100).detach().cpu().item()}


def learned_adaptive_mutual(node:Node, recorder, cur_round, args):
    """
    Learning for Ensemble
    @param node:
    @param recorder:
    @param cur_round:
    @param args:
    @return:
    """
    node.private_model = node.private_model.to(node.device)
    node.private_model.train()
    node.shared_model = node.shared_model.to(node.device)
    node.shared_model.train()
    if args.algorithm == 'equal_training':
        weight_private = 0.5
    else:
        if len(node.staged_learned_weight_inference) == 0:
            weight_private = 0.5
        else:
            weight_private = node.staged_learned_weight_inference[-1]
    print('Node {0}, ensemble training, training private weight: {1}'.format(node.idx, weight_private))
    train_loader = node.train_loader
    description = 'Node{:d}: Local Epoch {:d}, loss_private={:.4f} acc_private={:.2f}% loss_shared={:.4f} acc_shared={:.2f}%'
    for epoch in range(node.args.local_epoch):
        total_loss_private = 0.0
        correct_private = 0.0
        total_loss_shared = 0.0
        correct_shared = 0.0
        num_trained = 0
        with tqdm(train_loader) as bar_epoch:
            for idx, (data, target) in enumerate(bar_epoch):
                node.private_optimizer.zero_grad()
                node.shared_optimizer.zero_grad()
                data, target = data.to(node.device), target.to(node.device)
                output_private = node.private_model(data)
                output_shared = node.shared_model(data)
                ensemble_output_for_private = weight_private * output_private + (1 - weight_private) * output_shared.detach()
                ensemble_output_for_shared = weight_private * output_private.detach() + (1 - weight_private) * output_shared

                ce_private = CE_Loss(output_private, target)
                kl_private = KL_Loss(LogSoftmax(output_private), Softmax(output_shared.detach()))
                ce_shared = CE_Loss(output_shared, target)
                kl_shared = KL_Loss(LogSoftmax(output_shared), Softmax(output_private.detach()))
                        
                loss_private = ce_private + kl_private + CE_Loss(ensemble_output_for_private, target)  # the multiplication is to keep learning rate consistent with the vanilla mutual learning
                loss_shared = ce_shared + kl_shared + CE_Loss(ensemble_output_for_shared, target)
                
                loss_private.backward()
                loss_shared.backward()
                node.private_optimizer.step()
                node.shared_optimizer.step()
                
                total_loss_private += loss_private
                pred_private = output_private.argmax(dim=1)
                correct_private += pred_private.eq(target.view_as(pred_private)).sum()
                
                total_loss_shared += loss_shared
                pred_shared = output_shared.argmax(dim=1)
                correct_shared += pred_shared.eq(target.view_as(pred_shared)).sum()
            
                num_trained += len(data)

                bar_epoch.set_description(description.format(node.idx, epoch + 1, total_loss_private / (idx + 1), correct_private / num_trained * 100, total_loss_shared / (idx + 1), correct_shared / num_trained * 100))
    node.private_model = node.private_model.cpu()
    node.shared_model = node.shared_model.cpu()
    # save model
    if node.args.model_save == 'verbose':
        torch.save(node.private_model.state_dict(), os.path.join(args.log_root, str(recorder.time_stamp), 'models',
                                                                 'Node{0}_private_round{1}_{2}.pt'.format(node.idx,
                                                                                                          cur_round,
                                                                                                          node.args.private_model)))
        torch.save(node.shared_model.state_dict(), os.path.join(args.log_root, str(recorder.time_stamp), 'models',
                                                               'Node{0}_shared_round{1}_{2}.pt'.format(node.idx,
                                                                                                      cur_round,
                                                                                                      node.args.shared_model)))
    return {"idx": node.idx,
            "loss_private": (total_loss_private / (idx + 1)).detach().cpu().item(),
            "acc_private": (correct_private / num_trained * 100).detach().cpu().item(),
            "loss_shared": (total_loss_shared / (idx + 1)).detach().cpu().item(),
            "acc_shared": (correct_shared / num_trained * 100).detach().cpu().item()}    


def mutual(node: Node, recorder, cur_round, args):
    node.private_model = node.private_model.to(node.device)
    node.private_model.train()
    node.shared_model = node.shared_model.to(node.device)
    node.shared_model.train()

    train_loader = node.train_loader
    description = 'Node{:d}: Local Epoch {:d}, loss_private={:.4f} acc_private={:.2f}% loss_shared={:.4f} acc_shared={:.2f}%'
    for epoch in range(node.args.local_epoch):
        total_loss_private = 0.0
        correct_private = 0.0
        total_loss_shared = 0.0
        correct_shared = 0.0
        num_trained = 0
        with tqdm(train_loader) as bar_epoch:
            for idx, (data, target) in enumerate(bar_epoch):
                node.private_optimizer.zero_grad()
                node.shared_optimizer.zero_grad()
                data, target = data.to(node.device), target.to(node.device)
                output_private = node.private_model(data)
                output_shared = node.shared_model(data)
                ce_private = CE_Loss(output_private, target)
                kl_private = KL_Loss(LogSoftmax(output_private), Softmax(output_shared.detach()))
                ce_shared = CE_Loss(output_shared, target)
                kl_shared = KL_Loss(LogSoftmax(output_shared), Softmax(output_private.detach()))
                loss_private = ce_private + kl_private
                loss_shared = ce_shared + kl_shared
                loss_private.backward()
                loss_shared.backward()
                node.private_optimizer.step()
                node.shared_optimizer.step()
                
                total_loss_private += loss_private
                pred_private = output_private.argmax(dim=1)
                correct_private += pred_private.eq(target.view_as(pred_private)).sum()
                
                total_loss_shared += loss_shared
                pred_shared = output_shared.argmax(dim=1)
                correct_shared += pred_shared.eq(target.view_as(pred_shared)).sum()
            
                num_trained += len(data)

                bar_epoch.set_description(description.format(node.idx, epoch + 1, total_loss_private / (idx + 1), correct_private / num_trained * 100, total_loss_shared / (idx + 1), correct_shared / num_trained * 100))
                
    node.private_model = node.private_model.cpu()
    node.shared_model = node.shared_model.cpu()
    
    # save model
    if node.args.model_save == 'verbose':
        torch.save(node.private_model.state_dict(), os.path.join(args.log_root, str(recorder.time_stamp), 'models', 'Node{0}_private_round{1}_{2}.pt'.format(node.idx, cur_round, node.args.private_model)))
        torch.save(node.shared_model.state_dict(), os.path.join(args.log_root, str(recorder.time_stamp), 'models', 'Node{0}_shared_round{1}_{2}.pt'.format(node.idx, cur_round, node.args.shared_model)))
    
    return {"idx": node.idx, 
            "loss_private": (total_loss_private / (idx + 1)).detach().cpu().item(), 
            "acc_private": (correct_private / num_trained * 100).detach().cpu().item(), 
            "loss_shared": (total_loss_shared / (idx + 1)).detach().cpu().item(), 
            "acc_shared": (correct_shared / num_trained * 100).detach().cpu().item()}


def fed_avg(node: Node, recorder, cur_round, args):
    node.shared_model = node.shared_model.to(node.device)
    node.shared_model.train()
    train_loader = node.train_loader
    description = 'Node{:d}: Local Epoch {:d}, loss_shared={:.4f} acc_shared={:.2f}%'
    for epoch in range(node.args.local_epoch):
        total_loss_shared = 0.0
        correct_shared = 0.0
        num_trained = 0
        with tqdm(train_loader) as bar_epoch:
            for idx, (data, target) in enumerate(bar_epoch):
                node.shared_optimizer.zero_grad()
                data, target = data.to(node.device), target.to(node.device)
                output_shared = node.shared_model(data)
                ce_shared = CE_Loss(output_shared, target)
                ce_shared.backward()
                node.shared_optimizer.step()  
                total_loss_shared += ce_shared
                pred_shared = output_shared.argmax(dim=1)
                correct_shared += pred_shared.eq(target.view_as(pred_shared)).sum()
                num_trained += len(data)
                bar_epoch.set_description(description.format(node.idx, epoch + 1, total_loss_shared / (idx + 1), correct_shared / num_trained * 100))
    node.shared_model = node.shared_model.cpu()
    
    return {"idx": node.idx,
            "loss_shared": (total_loss_shared / (idx + 1)).detach().cpu().item(),
            "acc_shared": (correct_shared / num_trained * 100).detach().cpu().item()}


def individual(node: Node, recorder, cur_round, args):
    node.private_model = node.private_model.to(node.device)
    node.private_model.train()
    train_loader = node.train_loader
    description = 'Node{:d}: Local Epoch {:d}, loss_private={:.4f} acc_private={:.2f}%'
    for epoch in range(node.args.local_epoch):
        total_loss_private = 0.0
        correct_private = 0.0
        num_trained = 0
        with tqdm(train_loader) as bar_epoch:
            for idx, (data, target) in enumerate(bar_epoch):
                node.private_optimizer.zero_grad()
                data, target = data.to(node.device), target.to(node.device)
                output_private = node.private_model(data)
                ce_private = CE_Loss(output_private, target)
                ce_private.backward()
                node.private_optimizer.step()  
                total_loss_private += ce_private
                pred_shared = output_private.argmax(dim=1)
                correct_private += pred_shared.eq(target.view_as(pred_shared)).sum()
                num_trained += len(data)
                bar_epoch.set_description(description.format(node.idx, epoch + 1, total_loss_private / (idx + 1), correct_private / num_trained * 100))
    node.private_model = node.private_model.cpu()
    
    return {"idx": node.idx,
            "loss_private": (total_loss_private / (idx + 1)).detach().cpu().item(),
            "acc_private": (correct_private / num_trained * 100).detach().cpu().item()}


class Trainer(object):
    def __init__(self, recorder, args):
        self.args = args
        self.recorder = recorder
        if args.algorithm.lower() in ['learned_adaptive', 'mutual']:
            self.train = mutual
        elif args.algorithm.lower() in ['learned_adaptive_training', 'equal_training']:
            self.train = learned_adaptive_mutual
        elif args.algorithm.lower() in ['fed_avg', 'fed_avg_tune']:
            self.train = fed_avg
        elif args.algorithm.lower() == 'individual':
            self.train = individual
        elif args.algorithm.lower() == 'apfl':
            self.train = apfl
        else:
            raise AttributeError()

    def __call__(self, node, cur_round):
        record = self.train(node, self.recorder, cur_round, self.args)
        if self.recorder is not None:
            self.recorder.add_train_record(record)
