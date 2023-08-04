#!/usr/bin/env python

from __future__ import print_function
import os
import cv2
import shutil
import subprocess
import numpy as np
from collections import OrderedDict

import rospy


def convert_str_to_index(class_dict):
    index_class = {}
    for key, value in class_dict.items():
        index_class[value] = key
    return index_class

    
def make_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def result_display(img, save_path=''):
    if save_path is not '':
        cv2.imwrite(save_path, img)
        return 0
    cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
    cv2.imshow('Result', img)
    return cv2.waitKey(1)


def get_files_from_folder(base_path, valid_ext=["jpg", "jpeg", "png"]):
    file_lists = []
    for folder, _, files in os.walk(base_path):
        file_list = []
        folder = folder.replace('\\', '/')
        sub_dir = folder.replace(base_path, '')
        for filename in files:
            if filename.split('.')[-1] in valid_ext:
                filename_with_path = sub_dir + '/' + filename
                filename_with_path = filename_with_path[1:]
                file_list.append(filename_with_path)

        if len(file_list) > 0:
            file_lists.extend(file_list)
            print(" -%s -> Find %d data." % (folder, len(file_list)))

    print("# Total Data:", len(file_lists))
    return file_lists


def backup_code(output_dir):
    def _copy(output_dir, copy_dir, root='.'):
        if root.find('/model') != -1 or root.find('/tmp') != -1 or root.find('/log') != -1 or root.find('/result') != -1:
            return
        rootlist = os.listdir(root)
        if not os.path.exists(copy_dir):
            os.makedirs(copy_dir)
        for d in rootlist:
            fd_path = os.path.join(root, d)

            if os.path.isfile(fd_path):
                if '.zip' not in d and '.ckpt' not in d and '.npy' not in d and '.pyc' not in d \
                        and '.so' not in d and '_mask.c' not in d and 'evaluate_object' not in d and '.gitignore' not in d:
                    shutil.copy(fd_path, os.path.join(copy_dir, d))
            elif os.path.dirname(output_dir) not in fd_path:
                _copy(output_dir, os.path.join(copy_dir, d), fd_path)

    if os.path.exists(output_dir):
        subprocess.check_call(['rm', '-r', output_dir])

    _copy(output_dir, os.path.join(output_dir, 'code'))

def print_msg(f, msg):
    print(msg)
    print(msg, file=f)


def save_proc_time(proc_times, out_path, task_name='Unified'):
    if len(proc_times) <= 0:
        return

    proc_times = np.array(proc_times if len(proc_times) <= 10 else proc_times[10:])
    eval_list = OrderedDict()
    eval_list['Task_name'] = task_name
    eval_list['time_sample'] = proc_times.size
    eval_list['time_mean'] = np.mean(proc_times)
    eval_list['time_std'] = np.std(proc_times, ddof=1)
    eval_list['time_min'] = np.min(proc_times)
    eval_list['time_max'] = np.max(proc_times)

    with open(os.path.join(out_path, 'processing_time.log'), 'a') as f:
        for key in eval_list.keys():
            print_msg(f, key + " : " + str(eval_list[key]))
        f.write('\n')
