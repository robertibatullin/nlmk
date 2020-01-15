#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

classes = ['0-10','10-20','20-40','40-70']
colors = [(0,0,255), (0,200,255), (0,255,0), (255,0,0)]

data_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
    ])

random_crop = transforms.RandomCrop(32)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=0, stride=1) #(n,1,32,32) -> (n,32,28,28)
        self.pool1 = nn.MaxPool2d(kernel_size=2, padding=0, stride=2) #(n, 32,28,28) -> (n,32,14,14)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=3, padding=1, stride=1) #(n,32,14,14) -> (n,48,14,14)
        self.pool2 = nn.MaxPool2d(kernel_size=2, padding=1, stride=2) #(n, 48,14,14) -> (n,48,8,8)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=3, padding=1, stride=1) #(n,48,8,8) -> (n,64,8,8)
        self.pool3 = nn.MaxPool2d(kernel_size=3, padding=1, stride=1) #(n, 64,8,8) -> (n,64,8,8)
        self.fc1 = nn.Linear(64*8*8, 128) #(n, 64*8*8) -> (n,128)
        self.fc2 = nn.Linear(128, 4)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = x.view(x.size(0), 64*8*8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.tanh(x)
        return x

class FractionClassifier():
    
    def __init__(self, model_path):
        self.net = Net()
        self.net = self.net.float()
        self.net.load_state_dict(torch.load(model_path))
        self.net.eval()
        
    def get_fraction(self, arr, n_crops=5):
        crops = []
        img = Image.fromarray(arr)
        for i in range(n_crops):
            crops.append(random_crop(img))
        crops = torch.stack([data_transform(crop) for crop in crops])    
        inp = crops.float()
        out = self.net(inp)
        predict_proba = (out.squeeze()*0.5+0.5).detach().numpy()
        pp_mean = predict_proba.mean(axis=0)
        argmx = np.argmax(pp_mean)
        return classes[argmx], pp_mean[argmx]
    
    def process_video(self, path, 
                  start_time = 0,
                  end_time = None,
                  to_print = True,
                  to_show = False,
                  output_path = None):
    
        assert start_time >= 0
        assert end_time is None or end_time > start_time
        
        fractions = []
        probs = []
        times = []
    
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time*1000)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if to_show:
            cv2.namedWindow('Classification result', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Classification result", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        to_write = (output_path is not None)
        if to_write:
            ext = output_path.split('.')[-1]
            if ext != 'avi':
                output_path = output_path.replace(ext, 'avi')
            out = cv2.VideoWriter(output_path,
                                  cv2.VideoWriter_fourcc(*'DIVX'), 
                                  fps, (w,h))
        crop_top, crop_bottom = h//4, h//2
        crop_left, crop_right = w//4, w - w//4
        text_bottom = crop_bottom-4
        text_width = (crop_right - crop_left-4)//len(classes)
        text_lefts = [crop_left+4 + i*text_width for i in range(len(classes))]
    
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is False:
                break
            else:
                time = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                if end_time is None or time <= end_time:
                    roi = frame[crop_top:crop_bottom,
                                crop_left:crop_right:, :]
                    fraction, proba = self.get_fraction(roi)
                    fractions.append(fraction)
                    probs.append(proba)
                    times.append(time)

                    if to_print:
                        print( f'Time: {time:.2f} s\tFraction: {fraction}\tProbability: {proba:.3f}' )
                    if to_show or to_write:
                        display = frame
                        idx = classes.index(fraction)
                        cv2.rectangle(display, (crop_left, crop_top),
                                      (crop_right, crop_bottom),
                                      colors[idx], 2 )
                        cv2.putText(display, fraction,
                                    (text_lefts[idx], text_bottom),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    2, colors[idx], 4)
                        if to_write:
                            out.write(display)
                        else:
                            cv2.putText(display, 'Press Esc to quit',
                                        (crop_left, crop_top-4),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1, colors[idx], 2)
                            cv2.imshow('Classification result', display)
                            k = cv2.waitKey(1)
                            if k == 27:
                                break
                else:
                    break
        cap.release()
        if to_show:
            cv2.destroyAllWindows()
        if to_write:
            out.release()            
        return times, fractions, probs
       
