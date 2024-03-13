#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   datasets.py
@Time    :   2024-01-17 13:55:32
@Author  :   feng li
@Contact :   feng.li@tum.de
@Description    :   
'''

import numpy as np
import os
import torch
import cv2
import torch.utils.data as data
import matplotlib.pyplot as plt
from os import path

import core.tools as tools

class FreehandDataset(data.Dataset):
    def __init__(self, args, train=True):
        if train:
            self.img_dir = args.root_dir + '/image/train'
        else:
            self.img_dir = args.root_dir + '/image/val'
        self.pose_dir = args.root_dir + '/pose'
        self.others_dir = args.root_dir + '/others'

        self.neighbour_slice = args.neighbour_slice
        self.device = torch.device("cuda:{}".format(args.gpus[0]))

        samples = tools.filename_list(self.img_dir)
        self.samples = samples

        self.normalize = True

        self.tra_index_weight = args.t_weight
        self.rot_index_weight = args.r_weight

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        case_folder = self.samples[idx]
        case_id = int(case_folder[-4:])
        flow_dir = path.join(self.img_dir, 'Case{:04}'.format(case_id), 'flow')
        us_dir = path.join(self.img_dir, 'Case{:04}'.format(case_id), 'us')
        us_calib_mat = np.loadtxt(path.join(self.others_dir, 'freehand1.txt'))
        cam_calib_mat = np.loadtxt(path.join(self.others_dir, 'cam_calibration1.txt'))
        normalization = np.loadtxt(path.join(self.others_dir, 'normalization.txt'))

        start_offset = 21
        end_offset = 49

        case_pose = np.loadtxt(path.join(self.pose_dir, 'Case{:04}.txt'.format(case_id)))
        case_pose = case_pose[start_offset: len(case_pose) - end_offset, :]
        case_pose[:, 2:5] = case_pose[:, 2:5] * 1000

        frame_num = len(os.listdir(flow_dir)) - start_offset - end_offset

        files = os.listdir(flow_dir)
        files = sorted(files)
        file_name = files[start_offset - 1]
        # file_name = files[0]
        first_num = int(file_name[-8:-4])
        
        sample_size = self.neighbour_slice
        
        select_ids = tools.sample_ids(frame_num + 1, sample_size)
            
        us_sample_slices = []
        opf_sample_slices = [] # optical flow
        opf_nums = select_ids.shape[0] - 1
        labels = np.empty((opf_nums, 6))

        for i in range(select_ids.shape[0]):
            slice_index = select_ids[i]

            us_slice_path = path.join(us_dir, '{:04}.png'.format(slice_index + first_num))
            us_slice_img = cv2.imread(us_slice_path, 0)
            
            us_slice_img = tools.data_transform(us_slice_img, img_type='us')
            us_sample_slices.append(us_slice_img)
            
            if i != opf_nums:
                opf_slice_path = path.join(flow_dir, '{:04}.png'.format(slice_index + first_num))
                opf_slice_img = cv2.imread(opf_slice_path, 1)
                opf_slice_img = tools.data_transform(opf_slice_img, img_type='flow')
                opf_sample_slices.append(opf_slice_img)

                first_id = select_ids[i]
                second_id = select_ids[i + 1]
                # print(flow_dir)
                dof = tools.get_dof_label(case_pose[first_id, :], case_pose[second_id, :])
                if self.normalize:
                    dof = dof / normalization.flatten()
                labels[i] = dof

        us_sample_slices = np.asarray(us_sample_slices)
        opf_sample_slices = np.asarray(opf_sample_slices)

        us_sample_slices = np.expand_dims(us_sample_slices, axis=0)

        us_sample_slices = torch.from_numpy(us_sample_slices).float().to(self.device)
        opf_sample_slices = torch.from_numpy(opf_sample_slices).permute(3, 0, 1, 2).float().to(self.device)

        labels = torch.from_numpy(labels).float().to(self.device)

        return us_sample_slices, opf_sample_slices, labels
    
def data_loader(args, train=True):
    train_dataset = FreehandDataset(args, train=train)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size)
    return train_loader, len(train_dataset)