'''
demo for LiDAR-Iris
'''

import os
import sys
import cv2
import time
import argparse

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import sensor_msgs.point_cloud2 as pc2

from lidar_iris import one_couple_compare

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LiDAR-Iris demo')
    parser.add_argument('--data_path', type=str, default='./data', help='path to the data folder')
    # parser.add_argument('--data_name1', type=str, default='64line_1.pcd', help='name of cloud 1')
    # parser.add_argument('--data_name2', type=str, default='64line_2.pcd', help='name of cloud 2')
    parser.add_argument('--data_name1', type=str, default='32line_1.bin', help='name of cloud 1')
    parser.add_argument('--data_name2', type=str, default='32line_2.bin', help='name of cloud 2')    
    args = parser.parse_args()
    
    # load the data
    data_path1 = os.path.join(args.data_path, args.data_name1)
    data_path2 = os.path.join(args.data_path, args.data_name2)

    # process the data
    start_time = time.time()
    # load bin file
    cloud0 = np.fromfile(data_path1, dtype=np.float32)
    cloud0 = np.reshape(cloud0, (cloud0.shape[0]//4,4))
    cloud0 = cloud0[:,:3]
    cloud1 = np.fromfile(data_path2, dtype=np.float32)
    cloud1 = np.reshape(cloud1, (cloud1.shape[0]//4,4))
    cloud1 = cloud1[:,:3]
    print(cloud0.shape, cloud0)
    dis, bias = one_couple_compare(cloud0, cloud1)
    # dis, bias = one_couple_compare(data_path1, data_path2)
    end_time = time.time()
    print('description: dis = %.3f, bias = %d' % (dis, bias))
    print('time cost: %.3f s' % (end_time - start_time))
