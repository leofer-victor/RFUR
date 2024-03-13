#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test.py
@Time    :   2024-01-31 15:13:07
@Author  :   feng li
@Contact :   feng.li@tum.de
@Description    :   
'''

import argparse

import time
import matplotlib.pyplot as plt
import numpy as np
import cv2
from os import path, listdir

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import core.network as network
import core.tools as tools

class test():
    def __init__(self, args):
        self.neighbour_slice = args.neighbour_slice
        self.cur_path = path.dirname(__file__)
        self.result_path = path.join(self.cur_path, 'results')
        self.group_size = args.group_size

        self.batch_size = args.batch_size

        self.tra_index_weight = args.t_weight
        self.rot_index_weight = args.r_weight

        self.normalize = True
        self.normalization = np.loadtxt(path.join(args.root_dir, 'others', 'normalization.txt')) 
        
        self.model = nn.DataParallel(network.network(args), device_ids=args.gpus)
        self.model.load_state_dict(torch.load(path.join(self.cur_path, args.checkpoints_dir, args.model)))
        self.model = self.model.module
        self.device = 'cuda'
        self.model.to(self.device)
        self.model.eval()
        since1 = time.time()
        self.flow_slices_path, self.us_slices_path, self.case_pose, self.us_cali_mat = self.get_test_data_path(args)

        self.slice_num = len(self.flow_slices_path)
        self.slice_ids = np.linspace(0, self.slice_num - 1, self.slice_num).astype(np.uint32)

        end_slice_nums = self.slice_num - self.neighbour_slice + 1
        
        batch_groups, last_start_num, last_batch = self.divide_group(self.slice_num)
        
        dof_res = np.zeros((self.slice_num, 6))
        dof_index = np.zeros(self.slice_num)

        for idx, group in enumerate(batch_groups):
            since = time.time()
            batch_res = self.get_batch_res(group)
            batch_res = batch_res[0]
            time_elapsed = time.time() - since
            print('*' * 5 + '1 batch time: {:.3f}s'.format(time_elapsed % 60))
            # ## each frame
            for idy, pose in enumerate(batch_res):
                dof_res[idx * (args.group_size - 1) + idy] = pose
                dof_index[idx * (args.group_size - 1) + idy] += 1

        last_batch_res = self.get_batch_res(last_batch)
        last_batch_res = last_batch_res[0]

        idx = last_start_num

        for pose in last_batch_res:
            dof_res[idx] += pose
            dof_index[idx] += 1
            idx += 1

        for id in range(last_start_num, self.slice_num):
            if dof_index[id] > 1:
                dof_res[id] /= dof_index[id] 
        
        time_elapsed1 = time.time() - since1
        print('*' * 10 + 'Total time: {:.3f}s'.format(time_elapsed1 % 60))

        ## For slice group
        # for id, dof in enumerate(dof_res):
        #     if dof_index[id] > 1:
        #         dof /= dof_index[id] 
        
        dof_path = path.join(self.result_path, 'dof-{}.txt'.format(args.case))
        np.savetxt(dof_path, np.array(dof_res), fmt='%.8f')
        
        self.result = self.format_dof(dof_res)
        res_path = path.join(self.result_path, '{}.txt'.format(args.case))
        np.savetxt(res_path, np.array(self.result), fmt='%.8f')
        
        self.gt_pts = tools.params_to_corners(self.case_pose, self.us_cali_mat)
        self.res_pts = tools.params_to_corners(self.result, self.us_cali_mat)

        self.trans_pts1_error = tools.evaluate_dist(pts1=self.gt_pts[0 : len(self.res_pts), :, :], pts2=self.res_pts)
        self.final_drift = tools.final_drift(pts1=self.gt_pts[0 : len(self.res_pts), :, :], pts2=self.res_pts)

        print('{} distance error {:.4f}mm'.format(args.case, self.trans_pts1_error))
        print('{} final drift {:.4f}mm'.format(args.case, self.final_drift))
        print('*' * 50)

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')
        self.ax.set_box_aspect((2.5, 1, 1))  

        self.visualize_sequences(args)

        self.draw_dof_image(self.result, self.case_pose)

        # self.draw_accumulate_error(self.result, self.case_pose)

    def draw_dof_image(self, predictions, gt):
        nums = len(predictions)
        pred_dofs = np.empty((nums - 1, 6))
        gt_dofs = np.empty((nums - 1, 6))

        for i in range(nums - 1):
            pred_dof = tools.get_dof_label(predictions[i], predictions[i+1])
            gt_dof = tools.get_dof_label(gt[i], gt[i+1])
            pred_dofs[i] = pred_dof
            gt_dofs[i] = gt_dof

        x = np.linspace(1, nums - 1, nums - 1)
        for idx in range(0, 6):
            plt.subplot(2, 3, idx + 1)
            plt.plot(x, abs((pred_dofs[:, idx] - gt_dofs[:, idx]).flatten()), label='error')
            plt.plot(x, pred_dofs[:, idx].flatten(), label='prediction')
            plt.plot(x, gt_dofs[:, idx].flatten(), label='groundtruth')
            plt.legend(loc='upper right')
            if idx == 0:
                plt.title('tx (mm)')
            elif idx == 1:
                plt.title('ty (mm)')
            elif idx == 2:
                plt.title('tz (mm)')
            elif idx == 3:
                plt.title('rx (degree)')
            elif idx == 4:
                plt.title('ry (degree)')
            elif idx == 5:
                plt.title('rz (degree)')
        
        plt.show()

    def draw_accumulate_error(self, predictions, gt):
        nums = len(predictions)
        pred_dofs = np.empty((nums - 1, 6))
        gt_dofs = np.empty((nums - 1, 6))

        for i in range(nums - 1):
            pred_dof = tools.get_dof_label(predictions[i], predictions[i+1])
            gt_dof = tools.get_dof_label(gt[i], gt[i+1])
            pred_dofs[i] = pred_dof
            gt_dofs[i] = gt_dof

        x = np.linspace(1, nums - 1, nums - 1)
        
        res = np.empty((nums - 1, 6))
        for i in range(nums - 1):
            error = abs(pred_dofs[i, :] - gt_dofs[i, :])
            if i == 0:
                res[i, : ] = error
            else:
                res[i, :] = error + res[i - 1, :]

        for idx in range(0, 6):
            plt.subplot(2, 3, idx + 1)
            plt.plot(x, res[:, idx], label='accumulated error')
            plt.legend(loc='upper right')
            if idx == 0:
                plt.title('tx (mm)')
            elif idx == 1:
                plt.title('ty (mm)')
            elif idx == 2:
                plt.title('tz (mm)')
            elif idx == 3:
                plt.title('rx (degree)')
            elif idx == 4:
                plt.title('ry (degree)')
            elif idx == 5:
                plt.title('rz (degree)')
        
        plt.show()

    def draw_img_sequence(self, args, corner_pts):
        us_img_dir = path.join(args.root_dir, 'image', 'test', args.case, 'us')
        for frame_id in range(corner_pts.shape[0]):
            if frame_id % 3 == 0 or frame_id % 5 == 0 or frame_id % 7 == 0 or frame_id % 9 == 0:
                continue
            w_weights, h_weights = np.meshgrid(np.linspace(0, 1, 224), np.linspace(0, 1, 224))
            X = (1 - w_weights - h_weights) * corner_pts[frame_id, 0, 0] + \
                h_weights * corner_pts[frame_id, 3, 0] + w_weights * corner_pts[frame_id, 1, 0]
            Y = (1 - w_weights - h_weights) * corner_pts[frame_id, 0, 1] + \
                h_weights * corner_pts[frame_id, 3, 1] + w_weights * corner_pts[frame_id, 1, 1]
            Z = (1 - w_weights - h_weights) * corner_pts[frame_id, 0, 2] + \
                h_weights * corner_pts[frame_id, 3, 2] + w_weights * corner_pts[frame_id, 1, 2]

            img_path = path.join(us_img_dir, '{:04}.png'.format(frame_id))
            input_img = cv2.imread(img_path, 0)
            input_img = tools.data_transform(input_img)
            input_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2RGB)
            input_img = input_img / 255

            if frame_id == 0 or frame_id == corner_pts.shape[0] - 1:
                stride = 2
            else:
                stride = 10
            self.ax.plot_surface(X, Y, Z, rstride=stride, cstride=stride, facecolors=input_img, zorder=0.1)
    
    def draw_one_sequence(self, corner_pts, name, colorRGB=(255, 0, 0), line_width=1, constant=True):
        colorRGB = tuple(channel/255 for channel in colorRGB)
        seg_num = corner_pts.shape[0] + 1

        if constant:
            constant_color = np.asarray(colorRGB)
            constant_color = np.expand_dims(constant_color, axis=0)
            colors = np.repeat(constant_color, seg_num, axis=0)
        else:
            colors_R = np.linspace(0, colorRGB[0], seg_num).reshape((seg_num, 1))
            colors_G = np.linspace(0, colorRGB[1], seg_num).reshape((seg_num, 1))
            colors_B = np.linspace(1, colorRGB[2], seg_num).reshape((seg_num, 1))

            colors = np.concatenate((colors_R, colors_G, colors_B), axis=1)

        for frame_id in range(corner_pts.shape[0]):
            if frame_id == 0:
                """ First frame draw full bounds"""
                for pt_id in range(-1, 3):
                    xs = corner_pts[frame_id, pt_id, 0], corner_pts[frame_id, pt_id + 1, 0]
                    ys = corner_pts[frame_id, pt_id, 1], corner_pts[frame_id, pt_id + 1, 1]
                    zs = corner_pts[frame_id, pt_id, 2], corner_pts[frame_id, pt_id + 1, 2]
                    self.ax.plot(xs, ys, zs, color='b', lw=line_width, zorder=1)
            elif frame_id == corner_pts.shape[0] - 1:
                """ Connect to the former frame """
                for pt_id in range(-1, 3):
                    xs = corner_pts[frame_id, pt_id, 0], corner_pts[frame_id - 1, pt_id, 0]
                    ys = corner_pts[frame_id, pt_id, 1], corner_pts[frame_id - 1, pt_id, 1]
                    zs = corner_pts[frame_id, pt_id, 2], corner_pts[frame_id - 1, pt_id, 2]
                    self.ax.plot(xs, ys, zs, color=tuple(colors[frame_id, :]), lw=line_width)
                """ Last frame draw full bounds"""
                for pt_id in range(-1, 3):
                    xs = corner_pts[frame_id, pt_id, 0], corner_pts[frame_id, pt_id + 1, 0]
                    ys = corner_pts[frame_id, pt_id, 1], corner_pts[frame_id, pt_id + 1, 1]
                    zs = corner_pts[frame_id, pt_id, 2], corner_pts[frame_id, pt_id + 1, 2]
                    self.ax.plot(xs, ys, zs, color=tuple(colors[-1, :]), lw=line_width)
                    if pt_id == -1:
                        self.ax.plot(xs, ys, zs, color=tuple(colors[-1, :]), lw=line_width, label=name)
            else:
                """ Connect to the former frame """
                for pt_id in range(-1, 3):
                    xs = corner_pts[frame_id, pt_id, 0], corner_pts[frame_id - 1, pt_id, 0]
                    ys = corner_pts[frame_id, pt_id, 1], corner_pts[frame_id - 1, pt_id, 1]
                    zs = corner_pts[frame_id, pt_id, 2], corner_pts[frame_id - 1, pt_id, 2]
                    self.ax.plot(xs, ys, zs, color=tuple(colors[frame_id, :]), lw=line_width, zorder=1)

    def get_cpts(self, points):
        pts_all = np.empty((len(points), 3))
        boundary = np.empty((2, 3))
        x_min = 2000
        y_min = 2000
        z_min = 2000

        x_max = -2000
        y_max = -2000
        z_max = -2000
        for idx, pts in enumerate(points):
            pt = np.mean(pts, axis=0)
            if pt[0] < x_min:
                x_min = int(pt[0])
            if pt[1] < y_min:
                y_min = int(pt[1])
            if pt[2] < z_min:
                z_min = int(pt[2])

            if pt[0] > x_max:
                x_max = int(pt[0])
            if pt[1] > y_max:
                y_max = int(pt[1])
            if pt[2] > z_max:
                z_max = int(pt[2])
        
            pts_all[idx] = pt
        
        boundary = np.array([[x_min, y_min, z_min], [x_max, y_max, z_max]], dtype='int16')
        return pts_all, boundary

    def visualize_sequences(self, args):
        self.draw_one_sequence(corner_pts=self.gt_pts, name='Groundtruth', colorRGB=(255, 0, 0))
        self.draw_one_sequence(corner_pts=self.res_pts, name='RFUR-Net ({:.4f}mm)'.format(self.trans_pts1_error), colorRGB=(0, 153, 76))

        pts_all, boundary = self.get_cpts(self.gt_pts)
        datax = pts_all[:, 0]
        datay = pts_all[:, 1]
        dataz = pts_all[:, 2]
        self.ax.plot(datax, datay, dataz, c='y',marker='',linestyle='--')

        plt.axis('on')
        self.ax.set_xticks(np.linspace(boundary[0][0] - 30, boundary[1][0] + 30, 10))
        self.ax.set_yticks(np.linspace(boundary[0][1] - 40, boundary[1][1] + 40, 5))
        self.ax.set_zticks(np.linspace(boundary[0][2] - 30, boundary[1][2] + 30, 5))
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")

        plt.legend(loc='lower left')
        plt.tight_layout()

        self.ax.view_init(elev=45., azim=60)

        plt.title(args.case)
        plt.savefig(path.join(self.result_path, 'plots', '{}_visual.jpg'.format(args.case)))
        plt.show()

    def format_dof(self, format_dofs):
        format_res = []
        for i in range(len(format_dofs)):
            if i == 0:
                base_param = np.zeros(6)
                base_param = self.case_pose[i, 2:]
            else:
                base_param = format_res[i-1]
            gen_dof = format_dofs[i, :]
            gen_param = tools.get_next_pose(base_param, gen_dof, self.us_cali_mat)
            format_res.append(gen_param)
        format_res = np.asarray(format_res)
        pos_params = np.zeros((len(format_dofs) + 1, 7))
        pos_params[0, :] = self.case_pose[0, 2:]
        pos_params[1:, :] = format_res
        return pos_params
        
    def get_batch_res(self, group):
        flow_slices = []
        us_slices = []

        for id in range(len(group)):
            flow_img = cv2.imread(self.flow_slices_path[group[id]], 1)
            us_img = cv2.imread(self.us_slices_path[group[id]], 0)
            flow_img = tools.data_transform(flow_img, img_type='flow')
            us_img = tools.data_transform(us_img, img_type='us')
            flow_slices.append(flow_img)
            us_slices.append(us_img)

        # Add last ultrasound image
        us_img = cv2.imread(self.us_slices_path[int(group[-1])], 0)
        us_img = tools.data_transform(us_img, img_type='us')
        us_slices.append(us_img)

        # plt.figure()
        # plt.imshow(us_slices[0]) 
        # plt.show()

        flow_slices = np.asarray(flow_slices)
        us_slices = np.asarray(us_slices)

        # Ultrasound img has one less dimension than flow img because of channel equals to one
        us_slices = np.expand_dims(us_slices, axis=0)

        # Test datasets have no batch
        flow_slices = np.expand_dims(flow_slices, axis=0)
        us_slices = np.expand_dims(us_slices, axis=0)
        
        flow_slices = torch.from_numpy(flow_slices).permute(0, 4, 1, 2, 3).float().to(self.device)
        us_slices = torch.from_numpy(us_slices).float().to(self.device)

        trans, quat = self.model(us_slices, flow_slices, args.neighbour_slice)

        outputs = torch.cat((trans, quat), 2)

        outputs = outputs.data.cpu().numpy()
        if self.normalize:
            outputs = outputs * self.normalization.flatten()
        return outputs
    
    def get_tf_index(self, dof_global):
        tf_index = np.empty(6)
        i = 0
        for value in dof_global[0:3]:
            if abs(value) < self.tra_index_weight:
                tf_index[i] = 0
            else:
                tf_index[i] = 1
            i += 1
        for value in dof_global[3:]:
            if abs(value) < self.rot_index_weight:
                tf_index[i] = 0
            else:
                tf_index[i] = 1
            i += 1

        return tf_index

    def get_test_data_path(self, args):
        flow_img_dir = path.join(args.root_dir, 'image', 'test', args.case, 'flow')
        us_img_dir = path.join(args.root_dir, 'image', 'test', args.case, 'us')
        flow_imgs_name = listdir(flow_img_dir)
        us_imgs_name = listdir(us_img_dir)
        flow_imgs_name = sorted(flow_imgs_name)
        us_imgs_name = sorted(us_imgs_name)
        pose_dir = path.join(args.root_dir, 'pose')

        flow_slices_path = []
        us_slices_path = []
        for flow_name in flow_imgs_name:
            flow_slices_path.append(path.join(flow_img_dir, flow_name))
        for us_name in us_imgs_name:
            us_slices_path.append(path.join(us_img_dir, us_name)) 

        case_pose = np.loadtxt(path.join(pose_dir, '{}.txt'.format(args.case)))
        case_pose[:, 2:5] = case_pose[:, 2:5] * 1000
        us_cali_mat = np.loadtxt(path.join(args.root_dir, 'others', 'freehand1.txt'))

        return flow_slices_path, us_slices_path, case_pose, us_cali_mat
    
    def divide_group(self, slice_num):
        groups_num = slice_num // (self.group_size - 1) 
        last_batch_size = slice_num - 1 % self.group_size

        last_start_num = 0
        if last_batch_size != 0:
            last_start_num = slice_num - self.group_size + 1
 
        groups_ids = []
        for i in range(groups_num):
            this_batch_id = self.slice_ids[i * (self.group_size - 1) : (i + 1) * (self.group_size - 1) + 1]
            groups_ids.append(this_batch_id)

        if last_batch_size != 0:
            last_batch = self.slice_ids[slice_num - self.group_size + 1 : slice_num]
            
        return groups_ids, last_start_num, last_batch
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='rfur', help="name your experiment")
    parser.add_argument('--neighbour_slice', type=int, help='number of slice that acts as one sample', default='8')
    parser.add_argument('--model', default='rfur', help="name your model")
    parser.add_argument('--case', type=str, default='Case0006')
    parser.add_argument('--group_size', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])

    parser.add_argument('--t_weight', type=float, default=0.1)
    parser.add_argument('--r_weight', type=float, default=0.01)
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints')
    parser.add_argument('--root_dir', default='/home/leofer/experiment/freehand/freehand')

    args = parser.parse_args()

    test(args)