#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 12:05:21 2020

@author: robert
"""
import os
import nlmk
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--daytime", type=str, default='day', help="day or night")
parser.add_argument("--model_folder", type=str, 
                    default='pth', 
                    help="path to folder with .pth")
parser.add_argument("--video_file", type=str, help="path to video file")
parser.add_argument("--video_folder", type=str, 
                    help="path to folder with videos")
parser.add_argument("--video_file_index", type=int, 
                    default=0, 
                    help="video file index")
parser.add_argument("--conf_thres", type=float, default=0.5, help="detector confidence threshold")
parser.add_argument("--stack_size", type=int, default=10, help="detector stack size")
parser.add_argument("--start", type=int, default=0, help="start time in write mode")
opt = parser.parse_args()
model = opt.daytime
if model == 'day':
    n_classes = 5
elif model == 'night':
    n_classes = 4
pth_path = os.path.join(opt.model_folder, model+'.pth')

fc = nlmk.FractionClassifier(pth_path, 
                             n_classes,
                             conf_threshold=opt.conf_thres,
                             stack_size=opt.stack_size)
if opt.video_file is not None:
    path = opt.video_file
elif opt.video_folder is not None and opt.video_file_index is not None:
    video_folder = opt.video_folder #os.path.join('.', model, opt.video_folder)
    videos = os.listdir(video_folder)
    videos = list(filter(lambda vid:os.path.isfile(os.path.join(video_folder, vid)),videos))
    idx = opt.video_file_index
    vid = videos[idx]
    path = os.path.join(video_folder, vid)
else:
    raise FileNotFoundError('Video file not specified')
start_time = opt.start
fc.process_video(path, 
                 start_time=start_time,
                 to_show=True,
                 to_print=False)
