import math


# decay the learning rate exponentially
def lr_schedule_exponential(cur_round, node_list: list, args):
    cur_lr = args.lr * math.pow(args.lr_decay, cur_round)
    for i in range(len(node_list)):
        node_list[i].private_optimizer.param_groups[0]['lr'] = cur_lr
        node_list[i].shared_optimizer.param_groups[0]['lr'] = cur_lr
    print('Learning rate={:.6f}'.format(node_list[0].private_optimizer.param_groups[0]['lr']))
