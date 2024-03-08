#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   tools.py
@Time    :   2024-01-17 13:55:24
@Author  :   feng li
@Contact :   feng.li@tum.de
@Description    :   
'''

import math
import os
import cv2
import numpy as np
import random

import matplotlib.pyplot as plt

from numpy.linalg import inv
from transforms3d import euler, quaternions
# import transformations as tfms

def filename_list(dir):
    images = []
    dir = os.path.expanduser(dir)
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        images.append(file_path)
    return images

def evaluate_dist(pts1, pts2, resolution=1):
    """
    points input formats are frame_num x 4 (corner_points) x 3 (xyz)
    """
    error = np.square(pts1 - pts2)
    error = np.sum(error, axis=2)
    error = np.sqrt(error)
    error = np.mean(error) * resolution
    return error

def final_drift(pts1, pts2, resolution=1):
    # print(pts1.shape, pts2.shape)
    center_pt1 = np.mean(pts1, axis=0)
    center_pt2 = np.mean(pts2, axis=0)
    dist = np.linalg.norm(center_pt1 - center_pt2) * resolution
    return dist

def params_to_corners(params, us_cali_mat, input_img=np.ones((45, 50)), shrink=1):
    """
    Transform the params to corner points coordinates of each frame
    """
    h, w = input_img.shape

    # Four coordinates of the corners, depending on your setup.
    corner_pts = np.asarray([[0, -w/2, -h],
                             [0, w/2, -h],
                             [0, w/2, 0],
                             [0, -w/2, 0]])

    corner_pts = np.concatenate((corner_pts, np.ones((4, 1))), axis=1)
    corner_pts = np.transpose(corner_pts)

    transformed_pts = []
    for frame_id in range(params.shape[0]):
        trans_mat = params_to_mat(trans_params=params[frame_id, :])
        trans_mat = np.dot(trans_mat, us_cali_mat)
        transformed_corner_pts = np.dot(trans_mat, corner_pts)

        transformed_corner_pts = np.moveaxis(transformed_corner_pts[:3, :], 0, 1)

        transformed_pts.append(transformed_corner_pts)
    transformed_pts = np.asarray(transformed_pts)
    return transformed_pts

# def get_next_pos(trans_params1, dof, us_cali_mat):
#     """
#     Transfer 6d pose to translation and quaternion
#     """
#     trans_mat1 = params_to_mat(trans_params1, us_cali_mat=us_cali_mat)

#     dof[3:] = dof[3:] * (2 * math.pi) / 360

#     rot_mat = euler.euler2mat(dof[5], dof[4], dof[3], 'rzyx')

#     relative_mat = np.identity(4)
#     relative_mat[:3, :3] = rot_mat
#     relative_mat[:3, 3] = dof[:3]

#     next_mat = np.dot(inv(us_cali_mat), inv(np.dot(relative_mat, trans_mat1)))

#     quat = quaternions.mat2quat(next_mat[:3, :3])

#     next_params = np.zeros(7)
#     next_params[:3] = next_mat[:3, 3]
#     next_params[3:6] = quat[1:]
#     next_params[6] = quat[0]

#     return next_params

def get_next_pose(trans_params1, dof, us_cali_mat):
    """
    Transfer 6d pose to translation and quaternion
    """
    trans_mat1 = params_to_mat(trans_params1)

    dof[3:] = dof[3:] * (2 * math.pi) / 360

    rot_mat = euler.euler2mat(dof[5], dof[4], dof[3], 'rzyx')

    relative_mat = np.identity(4)
    relative_mat[:3, :3] = rot_mat
    relative_mat[:3, 3] = dof[:3]

    next_mat = np.dot(trans_mat1, inv(relative_mat))

    quat = quaternions.mat2quat(next_mat[:3, :3])

    next_params = np.zeros(7)
    next_params[:3] = next_mat[:3, 3]
    next_params[3:6] = quat[1:]
    next_params[6] = quat[0]

    return next_params

def get_next_pose_7d(trans_params1, dof, cali_mat):
    """
    Transfer 6d pose to translation and quaternion
    """
    trans_mat1 = params_to_mat(trans_params1, cali_mat)

    rot_mat = quaternions.quat2mat(xyzw_to_wxyz(dof[3:]))
    relative_mat = np.identity(4)
    relative_mat[:3, :3] = rot_mat
    relative_mat[:3, 3] = dof[:3]

    next_mat = np.dot(trans_mat1, inv(relative_mat))
    # next_mat = np.dot(next_mat, us_cali_mat)

    quat = quaternions.mat2quat(next_mat[:3, :3])

    next_params = np.zeros(7)
    next_params[:3] = next_mat[:3, 3]
    next_params[3:6] = quat[1:]
    next_params[6] = quat[0]

    return next_params

# def params_to_mat(trans_params, us_cali_mat):
#     """
#     Transform the 7 parameters into 4 x 4 matrix
#     """
#     if trans_params.shape[0] == 9:
#         trans_params = trans_params[2:]

#     translation = trans_params[:3]
#     quaternion = trans_params[3:]

#     r_mat = quaternions.quat2mat(xyzw_to_wxyz(quaternion))

#     trans_mat = np.identity(4)
#     trans_mat[:3, :3] = r_mat
#     trans_mat[:3, 3] = translation

#     # trans_mat = np.dot(trans_mat, us_cali_mat)
#     trans_mat = np.dot(us_cali_mat, trans_mat)
#     trans_mat = inv(trans_mat)

#     return trans_mat

def params_to_mat(trans_params):
    if trans_params.shape[0] == 9:
        trans_params = trans_params[2:]

    translation = trans_params[:3]
    quaternion = trans_params[3:]
    r_mat = quaternions.quat2mat(xyzw_to_wxyz(quaternion))

    trans_mat = np.identity(4)
    trans_mat[:3, :3] = r_mat
    trans_mat[:3, 3] = translation

    # trans_mat = np.dot(trans_mat, us_cali_mat)

    return trans_mat

# def get_6dof_label(trans_params1, trans_params2, us_cali_mat):
#     trans_mat1 = params_to_mat(trans_params1, us_cali_mat=us_cali_mat)
#     trans_mat2 = params_to_mat(trans_params2, us_cali_mat=us_cali_mat)

#     relative_mat = np.dot(trans_mat2, inv(trans_mat1))

#     pose = np.empty(6)
#     trans = relative_mat[:3, 3]
#     pose[:3] = trans
#     rot_mat = relative_mat[:3, :3]
#     angle = euler.mat2euler(rot_mat)
#     pose[3:6] = angle
#     return pose

def get_dof_label(trans_params1, trans_params2):
    trans_mat1 = params_to_mat(trans_params1)
    trans_mat2 = params_to_mat(trans_params2)

    relative_mat = np.dot(inv(trans_mat2), trans_mat1)

    pose = np.empty(6)
    trans = relative_mat[:3, 3]
    pose[:3] = trans
    rot_mat = relative_mat[:3, :3]
    angle = euler.mat2euler(rot_mat)
    # quat = quaternions.mat2quat(rot_mat)
    pose[3:] = angle
    pose[3:] = deg(pose[3:])
    return pose

def get_max_min(pose):
    x_max = -1000
    y_max = -1000
    z_max = -1000
    x_min = 1000
    y_min = 1000
    z_min = 1000
    theta_x_max = -1000
    theta_y_max = -1000
    theta_z_max = -1000
    theta_x_min = 1000
    theta_y_min = 1000
    theta_z_min = 1000

    w_max = -1000
    w_min = 1000
    
    if pose[0] > x_max:
        x_max = pose[0]
    if pose[1] > y_max:
        y_max = pose[1]
    if pose[2] > z_max:
        z_max = pose[2]

    if pose[0] < x_min:
        x_min = pose[0]
    if pose[1] < y_min:
        y_min = pose[1]
    if pose[2] < z_min:
        z_min = pose[2]
    
    if pose[3] > theta_x_max:
        theta_x_max = pose[3]
    if pose[4] > theta_y_max:
        theta_y_max = pose[4]
    if pose[5] > theta_z_max:
        theta_z_max = pose[5]

    if pose[3] < theta_x_min:
        theta_x_min = pose[3]
    if pose[4] < theta_y_min:
        theta_y_min = pose[4]
    if pose[5] < theta_z_min:
        theta_z_min = pose[5]
    
    if len(pose) == 7:
        if pose[6] > w_max:
            w_max = pose[6]
        
        if pose[6] < w_min:
            w_min = pose[6]
    
    # max = [x_max, y_max, z_max, theta_x_max, theta_y_max, theta_z_max, w_max]
    # min = [x_min, y_min, z_min, theta_x_min, theta_y_min, theta_z_min, w_min]

    max = [x_max, y_max, z_max, theta_x_max, theta_y_max, theta_z_max]
    min = [x_min, y_min, z_min, theta_x_min, theta_y_min, theta_z_min]
    return max, min

def data_transform(input_img, img_type='us', crop_size=672, resize=224):
    """
    Crop and resize image.
    """
    if img_type == 'us':
        h, w = input_img.shape
        x_start = int((h - crop_size) / 2)
        y_start = int((w - crop_size) / 2)
        patch_img = input_img[x_start:x_start+crop_size, y_start:y_start+crop_size]
        patch_img = cv2.resize(patch_img, (resize, resize))
    elif img_type == 'flow':
        h, w, c = input_img.shape
        x_start = int((h - crop_size) / 2)
        y_start = int((w - crop_size) / 2)
        # patch_img = input_img[x_start:x_start+crop_size, y_start:y_start+crop_size, :]
        patch_img = cv2.resize(input_img, (resize, resize))
        
    elif img_type == 'rgb':
        h, w = input_img.shape
        x_start = int((h - crop_size) / 2)
        y_start = int((w - crop_size) / 2)
        patch_img = input_img[x_start:x_start+crop_size, y_start:y_start+crop_size]
        patch_img = cv2.resize(patch_img, (resize, resize))
    return patch_img

def xyzw_to_wxyz(qua):
    if len(qua) != 4:
        return
    return np.append([qua[3]], qua[: 3])

def wxyz_to_xyzw(qua):
    if len(qua) != 4:
        return
    return np.append(qua[1:], [qua[0]])

def rad(degree):
    return degree * math.pi / 180

def deg(radian):
    return radian * 180 / math.pi

def sample_ids(slice_num, neighbour_num):
    start_range = slice_num - neighbour_num
    if start_range == 0:
        start_id = 0
    else:
        start_id = np.random.randint(0, start_range, 1)[0]
    select_ids = np.linspace(start_id, start_id + neighbour_num - 1, neighbour_num)
    select_ids = select_ids.astype(np.int64)
    return select_ids