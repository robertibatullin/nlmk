#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 12:05:21 2020

@author: robert
"""
import os
import nlmk
#from subprocess import check_call
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--daytime", type=str, default='night', help="day or night")
parser.add_argument("--model_folder", type=str, 
                    default='/home/robert/projects/celado/nlmk/model/v1/pth/', 
                    help="path to folder with .pth")
parser.add_argument("--n_classes", type=int, default=4, help="number of classes")
parser.add_argument("--video_folder", type=str, 
                    default='/home/robert/projects/celado/nlmk/model/v3/videos/test/night/0-10/',
                    help="path to folder with videos")
parser.add_argument("--output_folder", type=str, default='nlmk', 
                    help="folder to save output videos")
parser.add_argument("--video_file_index", type=int, 
                    default=0, 
                    help="video file index")
parser.add_argument("--conf_thres", type=float, default=0.5, help="detector confidence threshold")
parser.add_argument("--stack_size", type=int, default=10, help="detector stack size")
parser.add_argument("--write", type=int, default=0, help="write(1) or not(0) output file")
parser.add_argument("--start", type=int, default=0, help="start time in write mode")
#parser.add_argument("--delay", type=int, default=1, help="delay in write mode")
opt = parser.parse_args()
model = opt.daytime
pth_path = os.path.join(opt.model_folder, model+'.pth')

fc = nlmk.FractionClassifier(pth_path, 
                             opt.n_classes,
                             conf_threshold=opt.conf_thres,
                             stack_size=opt.stack_size)
videos = os.listdir(opt.video_folder)
videos = list(filter(lambda vid:os.path.isfile(os.path.join(opt.video_folder, vid)),
                     videos))
#idx, write, start = opt.video_file_index, opt.write, opt.start
idx, start = 2, 0
vid = videos[idx]
vid_name = '.'.join(vid.split('.')[:-1])
print(opt.video_folder, vid_name)
#if write == 0:
#    delay=1
output_path=None
start_time=start
#elif write == 1:
#    delay=opt.delay
#    output_path = vid_name+'.avi'
#    mp4_path = vid_name+'.mp4'
#    start_time=start
path = os.path.join(opt.video_folder, vid)
fc.process_video(path, 
                 start_time=start_time,
                 to_show=True,
                 output_path=output_path,
#                 delay=delay,
                 to_print=False)
#if write==1:
#    check_call(["ffmpeg","-y","-i",
#            output_path,"-s",
#            "640x480",mp4_path])
#    os.remove(output_path)
