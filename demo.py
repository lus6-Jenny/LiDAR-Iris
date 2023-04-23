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

from lidar_iris import LidarIris, one_couple_compare

# def one_couple_compare(cloudFilePath1: str, cloudFilePath2: str):
#     iris = LidarIris(4, 18, 1.6, 0.75, 50)

#     if not os.path.exists(cloudFilePath1):
#         print(f"{cloudFilePath1} does not exist")
#         return
#     if not os.path.exists(cloudFilePath2):
#         print(f"{cloudFilePath2} does not exist")
#         return
    
#     cloud0 = o3d.io.read_point_cloud(cloudFilePath1)
#     cloud1 = o3d.io.read_point_cloud(cloudFilePath2)

#     cloud0 = np.asarray(cloud0.points)
#     cloud1 = np.asarray(cloud1.points)
#     # print(f"cloud0.shape = {cloud0.shape}, cloud1.shape = {cloud1.shape}")
    
#     # convert to pcl format
#     # cloud0 = array_to_pcl(cloud0)
#     # cloud1 = array_to_pcl(cloud1)

#     li1 = iris.get_iris(cloud0)
#     li2 = iris.get_iris(cloud1)

#     # imshow
#     img_iris = np.vstack([li1, li2])
#     img_iris = cv2.normalize(img_iris, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#     # cv2.imshow("LiDAR Iris", img_iris)

#     fd1 = iris.get_feature(li1)
#     fd2 = iris.get_feature(li2)

#     bias = 0
#     dis, _ = iris.compare(fd1, fd2, bias=bias)

#     print(f"try compare:\n{cloudFilePath1}\n{cloudFilePath2}\n"
#           f"dis = {dis}, bias = {bias}")

#     img_iris = np.vstack([fd1.img, fd2.img])
#     img_iris = cv2.normalize(img_iris, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#     cv2.imshow("LiDAR Iris before transformation", img_iris)

#     temp = iris.circ_shift(fd1.img, bias, axis=0)
#     img_iris = np.vstack([temp, fd2.img])
#     img_iris = cv2.normalize(img_iris, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#     cv2.imshow("LiDAR Iris after transformation", img_iris)

#     img_T = np.hstack([fd1.T, fd2.T])
#     img_T = cv2.normalize(img_T, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#     cv2.imshow("LiDAR Iris Template", img_T)

#     cv2.waitKey(0)

#     return dis, bias


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
    dis, bias = one_couple_compare(data_path1, data_path2)
    end_time = time.time()
    print('description: dis = %.3f, bias = %d' % (dis, bias))
    print('time cost: %.3f s' % (end_time - start_time))
