import torch
from torch.autograd import Variable
import torch.nn as nn
from tqdm import tqdm
from utility.log import IteratorTimer
# import torchvision
import numpy as np
import time
import pickle
import cv2
import random
import math
from math import sin,cos,log,pow


def to_onehot(num_class, label, alpha):
    return torch.zeros((label.shape[0], num_class)).fill_(alpha).scatter_(1, label.unsqueeze(1), 1 - alpha)


def mixup(input, target, gamma):
    # target is onehot format!
    perm = torch.randperm(input.size(0))
    perm_input = input[perm]
    perm_target = target[perm]
    return input.mul_(gamma).add_(1 - gamma, perm_input), target.mul_(gamma).add_(1 - gamma, perm_target)


def clip_grad_norm_(parameters, max_grad):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p[1].grad is not None, parameters))
    max_grad = float(max_grad)

    for name, p in parameters:
        grad = p.grad.data.abs()
        if grad.isnan().any():
            ind = grad.isnan()
            p.grad.data[ind] = 0
            grad = p.grad.data.abs()
        if grad.isinf().any():
            ind = grad.isinf()
            p.grad.data[ind] = 0
            grad = p.grad.data.abs()
        if grad.max() > max_grad:
            ind = grad>max_grad
            p.grad.data[ind] = p.grad.data[ind]/grad[ind]*max_grad  # sign x val

def random_augmentation(x):
    # Rotate
    # x
    anglex = math.radians(random.uniform(-18, 18))
    Rx = torch.tensor([[1, 0, 0],
                       [0, cos(anglex), sin(anglex)],
                       [0, -sin(anglex), cos(anglex)]])
    Rx = Rx.transpose(0,1)
    # y
    angley = math.radians(random.uniform(-18, 18))
    Ry = torch.tensor([[cos(angley), 0, -sin(angley)],
                       [0, 1, 0],
                       [sin(angley), 0, cos(angley)]])
    Ry = Ry.transpose(0,1)
    # z
    anglez = math.radians(random.uniform(-18, 18))
    Rz = torch.tensor([[cos(anglez), sin(anglez), 0],
                       [-sin(anglez), cos(anglez), 0],
                       [0, 0, 1]])
    Rz = Rz.transpose(0,1)
    R_r = torch.matmul(Rz, torch.matmul(Ry, Rx))
    
    # Shear
    sh = random.uniform(-0.3, 0.3)
    R_s = torch.tensor([[1, sh, sh],
                      [sh, 1, sh],
                      [sh, sh, 1]])
    
    #print(x.size()) #N, C, T, V, M
    x = torch.matmul(torch.matmul(x.permute(0, 4, 2, 3, 1), R_r.cuda()), R_s.cuda()) # N, C, T, V, M -> N, M, T, V, C
    x = x.permute(0, 4, 2, 3, 1) # N, M, T, V, C -> N, C, T, V, M
    return x

def train_classifier(data_loader, model, loss_function, optimizer, global_step, args):
    loss_total = 0
    step = 0
    process = tqdm(IteratorTimer(data_loader), desc='Train: ')
    for index, (inputs, inputs1, labels) in enumerate(process):

        # label_onehot = to_onehot(args.class_num, labels, args.label_smoothing_num)
        # if args.mix_up_num > 0:
        #     # self.print_log('using mixup data: ', self.arg.mix_up_num)
        #     targets = to_onehot(args.class_num, labels, args.label_smoothing_num)
        #     inputs, targets = mixup(inputs, targets, np.random.beta(args.mix_up_num, args.mix_up_num))
        # elif args.label_smoothing_num != 0 or args.loss == 'cross_entropy_naive':
        #     targets = to_onehot(args.class_num, labels, args.label_smoothing_num)
        # else:
        #     targets = labels
        
        inputs, inputs1, labels = inputs.cuda(non_blocking=True), inputs1.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        
        #augmentation
        # aug1 = random_augmentation(inputs)
        # aug2 = random_augmentation(inputs)
        # aug12_input = torch.cat((aug1, aug2), dim=0)
        
        # aug3 = random_augmentation(inputs1)
        # aug4 = random_augmentation(inputs1)
        # aug34_input = torch.cat((aug3, aug4), dim=0)
        
        out1s, out1t, out2s, out2t, out11s, out11t, out12s, out12t, out1g, out2g, _, _ = model(inputs, inputs1) #joint, motion, joint*motion
        # out1s_1, out1s_2 = out1s.chunk(2, 0)
        # out1t_1, out1t_2 = out1t.chunk(2, 0)
        # out2s_1, out2s_2 = out2s.chunk(2, 0)
        # out2t_1, out2t_2 = out2t.chunk(2, 0)
        # out3s_1, out3s_2 = out3s.chunk(2, 0)
        # out3t_1, out3t_2 = out3t.chunk(2, 0)
        
        out1s = model.module.mlp(out1s)
        out1t = model.module.mlp(out1t)
        out2s = model.module.mlp(out2s)
        out2t = model.module.mlp(out2t)
        out11s = model.module.mlp(out11s)
        out11t = model.module.mlp(out11t)
        out12s = model.module.mlp(out12s)
        out12t = model.module.mlp(out12t)
        out1g = model.module.mlpg(out1g)
        out2g = model.module.mlpg(out2g)
        # out1s_1 = model.module.mlp(out1s_1)
        # out1t_1 = model.module.mlp(out1t_1)
        # out1s_2 = model.module.mlp(out1s_2)
        # out1t_2 = model.module.mlp(out1t_2)
        # out2s_1 = model.module.mlp(out2s_1)
        # out2t_1 = model.module.mlp(out2t_1)
        # out2s_2 = model.module.mlp(out2s_2)
        # out2t_2 = model.module.mlp(out2t_2)
        # out3s_1 = model.module.mlp(out3s_1)
        # out3t_1 = model.module.mlp(out3t_1)
        # out3s_2 = model.module.mlp(out3s_2) # N, V, C
        # out3t_2 = model.module.mlp(out3t_2) # N, T, C
        
        V = out1s.shape[1]
        T = out1t.shape[1]
        Ng = out1g.shape[0]
        loss = loss_function(torch.cat((out1s,out2s), dim=1), V) + loss_function(torch.cat((out1t,out2t), dim=1), T) \
              +loss_function(torch.cat((out1s,out12s), dim=1), V) + loss_function(torch.cat((out1t,out12t), dim=1), T) \
              +loss_function(torch.cat((out2s,out11s), dim=1), V) + loss_function(torch.cat((out2t,out11t), dim=1), T) \
              +loss_function(torch.cat((out11s,out12s), dim=1), V) + loss_function(torch.cat((out11t,out12t), dim=1), T) \
              +loss_function(torch.cat((out1g.unsqueeze(0), out2g.unsqueeze(0)), dim=1), Ng)
              # +loss_function(torch.cat((out2s_1,out2s_2), dim=1), V) + loss_function(torch.cat((out2t_1,out2t_2), dim=1), T) \
              # +loss_function(torch.cat((out3s_1,out3s_2), dim=1), V) + loss_function(torch.cat((out3t_1,out3t_2), dim=1), T)
        #loss = loss / 6.0
        
        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip:
            clip_grad_norm_(model.named_parameters(), args.grad_clip)
        optimizer.step()
        global_step += 1
        # if len(outputs.data.shape) == 3:  # T N cls
        #     _, predict_label = torch.max(outputs.data[:, :, :-1].mean(0), 1)
        # else:
        #     _, predict_label = torch.max(outputs.data, 1)
        #loss = loss_function(outputs, targets)
        ls = loss.data.item()
        #acc = torch.mean((predict_label == labels.data).float()).item()
        # ls = loss.data[0]
        # acc = torch.mean((predict_label == labels.data).float())
        loss_total += ls
        step += 1
        lr = optimizer.param_groups[0]['lr']
        process.set_description('loss: {:4f}, batch time: {:4f}, lr: {:4f}'.format(ls,process.iterable.last_duration,lr))

        # 每个batch记录一次
        # if args.mode == 'train_val':
        #     writer.add_scalar('acc', acc, global_step)
        #     writer.add_scalar('loss', ls, global_step)
        #     writer.add_scalar('batch_time', process.iterable.last_duration, global_step)
            # if len(inputs.shape) == 5:
            #     if index % 500 == 0:
            #         img = inputs.data.cpu().permute(2, 0, 1, 3, 4)
            #         # NCLHW->LNCHW
            #         img = torchvision.utils.make_grid(img[0::4, 0][0:4], normalize=True)
            #         writer.add_image('img', img, global_step=global_step)
            # elif len(inputs.shape) == 4:
            #     if index % 500 == 0:
            #         writer.add_image('img', ((inputs.cpu().numpy()[0] + 128) * 1).astype(np.uint8).transpose(1, 2, 0),
            #                          global_step=global_step)
    
    process.close()
    loss = loss_total / step
    return global_step, loss


def val_classifier(data_loader, model, loss_function, global_step, args):
    right_num_total = 0
    total_num = 0
    loss_total = 0
    step = 0
    process = tqdm(IteratorTimer(data_loader), desc='Val: ')
    # s = time.time()
    # t=0
    score_frag = []
    all_pre_true = []
    wrong_path_pre_ture = []
    for index, (inputs, inputs1, labels, path) in enumerate(process):
        # label_onehot = to_onehot(args.class_num, labels, args.label_smoothing_num)
        if args.loss == 'cross_entropy_naive':
            targets = to_onehot(args.class_num, labels, args.label_smoothing_num)
        else:
            targets = labels

        with torch.no_grad():
            inputs, targets, labels = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True), labels.cuda(
                non_blocking=True)
            outputs = model(inputs)
            if len(outputs.data.shape) == 3:  # T N cls
                _, predict_label = torch.max(outputs.data[:, :, :-1].mean(0), 1)
                score_frag.append(outputs.data.cpu().numpy().transpose(1,0,2))
            else:
                _, predict_label = torch.max(outputs.data, 1)
                score_frag.append(outputs.data.cpu().numpy())
            loss = loss_function(outputs, targets)

        predict = list(predict_label.cpu().numpy())
        true = list(labels.data.cpu().numpy())
        for i, x in enumerate(predict):
            all_pre_true.append(str(x) + ',' + str(true[i]) + '\n')
            if x != true[i]:
                wrong_path_pre_ture.append(str(path[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')

        right_num = torch.sum(predict_label == labels.data).item()
        # right_num = torch.sum(predict_label == labels.data)
        batch_num = labels.data.size(0)
        acc = right_num / batch_num
        ls = loss.data.item()
        # ls = loss.data[0]

        right_num_total += right_num
        total_num += batch_num
        loss_total += ls
        step += 1

        process.set_description(
            'Val-batch: acc: {:4f}, loss: {:4f}, time: {:4f}'.format(acc, ls, process.iterable.last_duration))
        # process.set_description_str(
        #     'Val: acc: {:4f}, loss: {:4f}, time: {:4f}'.format(t, t, t), refresh=False)
        # if len(inputs.shape) == 5:
        #     if index % 50 == 0 and (writer is not None) and args.mode == 'train_val':
        #         # NCLHW->LNCHW
        #         img = inputs.data.cpu().permute(2, 0, 1, 3, 4)
        #         img = torchvision.utils.make_grid(img[0::4, 0][0:4], normalize=True)
        #         writer.add_image('img', img, global_step=global_step)
        # elif len(inputs.shape) == 4:
        #     if index % 50 == 0 and (writer is not None) and args.mode == 'train_val':
        #         writer.add_image('img', ((inputs.cpu().numpy()[0] + 128) * 1).astype(np.uint8).transpose(1, 2, 0),
        #                          global_step=global_step)
    # t = time.time()-s
    # print('time: ', t)
    score = np.concatenate(score_frag)
    score_dict = dict(zip(data_loader.dataset.sample_name, score))

    process.close()
    loss = loss_total / step
    accuracy = right_num_total / total_num
    print('Accuracy: ', accuracy)
    # if args.mode == 'train_val' and writer is not None:
    #     writer.add_scalar('loss', loss, global_step)
    #     writer.add_scalar('acc', accuracy, global_step)
    #     writer.add_scalar('batch time', process.iterable.last_duration, global_step)

    return loss, accuracy, score_dict, all_pre_true, wrong_path_pre_ture

