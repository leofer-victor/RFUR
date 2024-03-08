#!/usr/bin/env python7
# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2024-01-17 13:54:17
@Author  :   feng li
@Contact :   feng.li@tum.de
@Description    :   
'''


import argparse
import time
from os import path, mkdir

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import core.network as network
import core.network_2d as network_2d
import core.datasets as datasets
import core.tools as tools
from core.logger import Logger

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

lowest_loss = 100
best_epoch = 0
saved_check_points_loss = 0.05

training_lowest_loss = 0.1

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def fetch_optimizer(args, model):
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.95)
    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate, total_steps=args.epochs + 10, pct_start=0.05, cycle_momentum=False)
    return optimizer, scheduler

def loss_function(criterion_mse, criterion_huber, trans, rot, labels, tf_index):
    tra_loss = 3 * criterion_mse(trans, labels[:, :, :3])
    rot_loss = 1 * criterion_mse(rot, labels[:, :, 3:])

    loss = tra_loss + rot_loss

    # print("trans:\n", trans)
    # print("quat:\n", rot)
    # print("gt:\n", labels, "\n")

    labels_sum = torch.sum(labels, dim=1)
    tra_sum_loss = criterion_huber(torch.sum(trans, dim=1), labels_sum[:, :3])
    rot_sum_loss = criterion_huber(torch.sum(rot, dim=1), labels_sum[:, 3:])
    sum_loss = 10 * (tra_sum_loss + rot_sum_loss)

    outputs = torch.cat((trans, rot), 2)

    # tra_corr_loss = 0.2 * get_correlation_loss(trans, labels[:, :, :3])
    # rot_corr_loss = 0.1 * get_correlation_loss(rot, labels[:, :, 3:])
    corr_loss = correlation_loss(trans, rot, labels)

    hybrid_loss = loss + sum_loss + corr_loss

    print("loss: ", loss, "sum_loss: ", sum_loss, "corr_loss: ", corr_loss)

    return hybrid_loss

def correlation_loss(trans, rot, labels):
    sum = 0
    for i in range(len(trans)):
        sum1 = get_correlation_loss_value(labels[i, :, 0], trans[i, :, 0]) + get_correlation_loss_value(labels[i, :, 1], trans[i, :, 1]) + get_correlation_loss_value(labels[i, :, 2], trans[i, :, 2]) \
            + get_correlation_loss_value(labels[i, :, 3], rot[i, :, 0]) + get_correlation_loss_value(labels[i, :, 4], rot[i, :, 1]) + get_correlation_loss_value(labels[i, :, 5], rot[i, :, 2])
        sum1 = sum1 / 6
        sum += sum1
    return 1 - sum

def get_correlation_loss_value(labels, outputs):
    x = outputs.flatten()
    y = labels.flatten()
    xy = x * y
    mean_xy = torch.mean(xy)
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    cov_xy = mean_xy - mean_x * mean_y
    var_x = torch.sum((x - mean_x) ** 2 / x.shape[0])
    var_y = torch.sum((y - mean_y) ** 2 / y.shape[0])
    corr_xy = cov_xy / (torch.sqrt(var_x * var_y))
    return corr_xy

@torch.no_grad()
def validate(criterion_mse, criterion_huber, epoch, model, val_dataset, val_dataset_size):
    model.eval()
    val_loss = 0.0

    for val_blob in val_dataset:
        us_sample_slices, opf_sample_slices, labels, tf_index = [x.cuda() for x in val_blob]
        # outputs = model(us_sample_slices, opf_sample_slices, args.neighbour_slice)
        trans, rot = model(us_sample_slices, opf_sample_slices, args.neighbour_slice)
        loss = loss_function(criterion_mse, criterion_huber, trans, rot, labels, tf_index)
        val_loss += loss.data.mean()

    epoch_loss = val_loss / val_dataset_size
    global lowest_loss
    global best_epoch
    if epoch_loss < lowest_loss and epoch > 200:
        lowest_loss = epoch_loss
        best_epoch = epoch + 1
        check_path = 'checkpoints/Best_val_%s.pth' % ('RFUR')
        torch.save(model.state_dict(), check_path)
        print('*' * 5 + 'Best model updated with loss={:.4f} at epoch {}.'.format(lowest_loss, epoch + 1))
    
    if epoch_loss < saved_check_points_loss and epoch > 400 and epoch_loss != lowest_loss:
        check_path = 'checkpoints/val_%d_%s_%f.pth' % (epoch + 1, 'rfur', epoch_loss)
        torch.save(model.state_dict(), check_path)
        print('*' * 5 + 'Check points with loss={:.4f} at epoch {}.'.format(epoch_loss, epoch + 1))

    return epoch_loss, lowest_loss, best_epoch

def train(args):
    model = nn.DataParallel(network.network(args), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))
    
    train_loader, train_dataset_size = datasets.data_loader(args)
    val_loader, val_dataset_size = datasets.data_loader(args, train=False)
    optimizer, scheduler = fetch_optimizer(args, model)
    criterion_mse = nn.MSELoss()
    criterion_huber = nn.HuberLoss(reduction='mean')

    logger = Logger(scheduler)
    logger.record(args)

    since = time.time()

    model.cuda()

    max_value = []
    min_value = []

    print('Training with %d.' % train_dataset_size)
    print('Validating with %d.' % val_dataset_size)
    for epoch in range(args.epochs):
        model.train()

        scaler = GradScaler()
        running_loss = 0.0

        for train_blob in train_loader:
            optimizer.zero_grad()
            us_sample_slices, opf_sample_slices, labels, tf_index = [x.cuda() for x in train_blob]
            
            ## Get max and min value of 6dof
            label_nor = labels.data.cpu().numpy()
            for i in range(label_nor.shape[0]):
                for j in range(label_nor.shape[1]):
                    dof_tmp = label_nor[i][j]
                    max_tmp, min_tmp = tools.get_max_min(dof_tmp)
                    if len(max_value) == 0:
                        max_value = max_tmp
                        min_value = min_tmp
                    else:
                        for k in range(len(max_value)):
                            if max_tmp[k] > max_value[k]:
                                max_value[k] = max_tmp[k]
                            if min_tmp[k] < min_value[k]:
                                min_value[k] = min_tmp[k]

            trans, rot = model(us_sample_slices, opf_sample_slices, args.neighbour_slice)
            loss = loss_function(criterion_mse, criterion_huber, trans, rot, labels, tf_index)
            running_loss += loss.data.mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)     
            scaler.step(optimizer)
            scaler.update()

        # print("max_value: ", max_value)
        # print("min_value: ", min_value)
        train_epoch_loss = running_loss / train_dataset_size

        global training_lowest_loss
        if train_epoch_loss < training_lowest_loss and epoch > args.epochs - 300:
            training_lowest_loss = train_epoch_loss
            best_epoch = epoch + 1
            check_path = 'checkpoints/Training_Best_%s_{}.pth'.format(epoch + 1) % ('RFUR')
            torch.save(model.state_dict(), check_path)
            print('*' * 5 + 'Best training model updated with loss={:.4f} at epoch {}.'.format(lowest_loss, epoch + 1))

        if train_epoch_loss < saved_check_points_loss and epoch > args.epochs - 100:
            check_path = 'checkpoints/tra_%d_%s_%f.pth' % (epoch + 1, 'rfur', train_epoch_loss)
            torch.save(model.state_dict(), check_path)
            print('*' * 5 + 'Check points with loss={:.4f} at epoch {}.'.format(train_epoch_loss, epoch + 1))

        # Validation
        val_epoch_loss, lowest_loss, best_epoch = validate(criterion_mse, criterion_huber, epoch, model, val_loader, val_dataset_size)

        scheduler.step()

        logger.update_record(best_epoch, epoch, lowest_loss)
        logger.print_training_status(epoch, args.epochs, best_epoch, train_epoch_loss, val_epoch_loss, lowest_loss)
        logger.write(train_epoch_loss, val_epoch_loss, epoch)

        model.train()
        

    time_elapsed = time.time() - since
    print('*' * 5 + 'Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('*' * 5 + 'Lowest validation loss: {:4f} at epoch {}'.format(lowest_loss, best_epoch))
    logger.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='rfur', help="name your experiment")
    parser.add_argument('--neighbour_slice', type=int, default='8')
    
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--epochs', type=int, help='number of training epochs', default=500)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])

    parser.add_argument('--t_weight', type=float, default=5)
    parser.add_argument('--r_weight', type=float, default=0.5)

    parser.add_argument('--root_dir', default='/home/leofer/experiment/freehand/freehand')

    args = parser.parse_args()

    device = torch.device("cuda:{}".format(args.gpus[0]))

    if not path.isdir('checkpoints'):
        mkdir('checkpoints')

    train(args)