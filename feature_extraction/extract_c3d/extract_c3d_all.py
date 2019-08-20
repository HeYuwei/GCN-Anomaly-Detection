# !/usr/bin/python
# -*- coding:utf8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import cv2
import numpy as np
from multiprocessing import Pool, current_process


# gpu_list = [0,1,2,3]
worker_cnt = 1

# score_name = "fc-action-ucf_crimes"
score_name = "relu6"
rgb_prefix = ""
# video_folder = "../ucf_crimes_rgb/"
video_folder = "/mmu_ssd/liuchang03/heyuwei/Data/crime_pic/Anomaly-Videos-Part-1/Abuse/"
modality = "c3d"
deploy_file = "./ucf_crimes/c3d_feature.prototxt"

caffe_path = "../caffe_c3d/"

# sys.path.append(os.path.join(caffe_path, "python"))
# sys.path.append('/usr/local/include/google/protobuf')
# sys.path.append('/usr/local/lib')
from pyActionRecog.action_caffe import CaffeNet

step = 16
dense_sample = True
# output_folder = "../c3d_features/"
output_folder = "/mmu_ssd/liuchang03/heyuwei/Data/crime_c3d_feature/"
caffemodel = "./models/c3d_iter_1000.caffemodel"


def build_net():
    # global net
    # gpu_id = gpu_list[current_process()._identity[0] % len(gpu_list)]
    gpu_id = 0
    net = CaffeNet(deploy_file, caffemodel, gpu_id)
    return net

def eval_video(video_frame_list):
    # global net
    net = build_net()
    print('net is loaded')

    for video_frame_path in video_frame_list:
        vid = os.path.basename(video_frame_path)
        print("video {} doing".format(vid))
        all_files = os.listdir(video_frame_path)
        frame_cnt = len(all_files)
        if modality == "c3d":
            stack_depth = 16
        else:
            raise ValueError(modality)
        # output_file = os.path.join(os.path.join(output_folder, os.path.basename(caffemodel).replace(".caffemodel","")), vid + "_c3d" + ".npz")
        output_file = os.path.join(output_folder, vid + "_c3d" + ".npz")
        if os.path.isfile(output_file):
            print("{} exists!".format(output_file))
            # return

        frame_ticks = range(1,frame_cnt+1, step)
        frame_scores = []
        for tick in frame_ticks:
            if modality == "c3d":
                if dense_sample:
                    frames = []
                    for i in range(0, step, stack_depth):
                        frame_idx = [min(frame_cnt, tick + i + offset) for offset in range(stack_depth)]
                        for idx in frame_idx:
                            name = "{}{:06d}.jpg".format(rgb_prefix, idx - 1)
                            # print('v name ' + os.path.join(video_frame_path, name))
                            frames.append(cv2.imread(os.path.join(video_frame_path, name), cv2.IMREAD_COLOR))
                    scores = net.predict_single_c3d_rgb_stack(frames, score_name, frame_size=(171,128))
                    print('feature shape ' + str(scores.shape))
                else:
                    print("Sparse sampling has yet to be done.")
            frame_scores.append(scores)
        np.savez(output_file, scores=frame_scores, begin_idx=frame_ticks)
        print("video {} done".format(vid))


if __name__ == '__main__':
    video_name_list = os.listdir(video_folder)
    video_path_list = [os.path.join(video_folder, it) for it in video_name_list]
    eval_video(video_path_list)
    # pool = Pool(processes=worker_cnt, initializer=build_net)
    # pool.map(eval_video, video_path_list)