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

classes = ['0-10','10-20','20-40','40-70', 'scrap']
colors = [(0,0,255), (0,200,255), (0,255,0), (255,0,0), (150,150,150)]
step = 20 #сдвиг окна в ответ на нажатие стрелки
original_size = {'w':1920, 'h':1080}

data_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
    ])

random_crop = transforms.RandomCrop(32)

class ROI():
    def __init__(self, rect):
        self.left, self.top, self.right, self.bottom = rect
        
    def size(self):
        return (self.right-self.left, self.bottom-self.top)
    
    def set_source(self, img):
        self.source = img
        self.img = img[self.top:self.bottom,
                       self.left:self.right:, :]

    def resize(self, new_size):
        self.img = cv2.resize(self.img, new_size)         

class Net(nn.Module):
    def __init__(self, n_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=0, stride=1) #(n,1,32,32) -> (n,32,28,28)
        self.pool1 = nn.MaxPool2d(kernel_size=2, padding=0, stride=2) #(n, 32,28,28) -> (n,32,14,14)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=3, padding=1, stride=1) #(n,32,14,14) -> (n,48,14,14)
        self.pool2 = nn.MaxPool2d(kernel_size=2, padding=1, stride=2) #(n, 48,14,14) -> (n,48,8,8)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=3, padding=1, stride=1) #(n,48,8,8) -> (n,64,8,8)
        self.pool3 = nn.MaxPool2d(kernel_size=3, padding=1, stride=1) #(n, 64,8,8) -> (n,64,8,8)
        self.fc1 = nn.Linear(64*8*8, 128) #(n, 64*8*8) -> (n,128)
        self.fc2 = nn.Linear(128, n_classes)
        
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
    
    def __init__(self, model_path, 
                 n_classes,
                 conf_threshold = 0.5,
                 stack_size = 5):
        self.n_classes = n_classes
        self.threshold = conf_threshold
        self.stack_size = stack_size
        self.stack = []
        self.net = Net(n_classes)
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
        pmax = pp_mean[argmx]
        if pmax >= self.threshold:
            return classes[argmx], pmax
        else:
            return '?', pmax
        
    def stack_vote(self): 
        #из N последних наблюдений в формате (фракция, вероятность)
        #выбираем фракцию с наибольшей суммой вероятностей
        #возвращаем эту фракцию и её среднюю вероятность
        arr = np.array(self.stack)
        groups = {c: arr[arr[:,0]==c][:,1].astype(float) \
                 for c in classes+['?']}
        sums = [ (c, groups[c].sum()) for c in groups]
        means = {c: groups[c].mean() for c in groups}
        sorted_sums = sorted(sums, key=lambda x:x[1])
        top_class = sorted_sums[-1][0]
        return top_class, means[top_class]
    
    def process_video(self, path, 
                  start_time = 0,
                  end_time = None,
                  to_print = True,
                  to_show = False,
                  output_path = None,
                  delay=1):
    
        assert start_time >= 0
        assert end_time is None or end_time > start_time
        
        fractions = []
        probs = []
        times = []
        
        try:
            cap = cv2.VideoCapture(path)
            cap.set(cv2.CAP_PROP_POS_MSEC, start_time*1000)
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f'w={w}, h={h}',file=open('log.log','w'))
            scale = original_size['w']/w
            
            if to_show:
                cv2.namedWindow('Classification result', cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty("Classification result", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            to_write = (output_path is not None)
            detect = not (to_show or to_write)
    
            if to_write:
                ext = output_path.split('.')[-1]
                if ext != 'avi':
                    output_path = output_path.replace(ext, 'avi')
                out = cv2.VideoWriter(output_path,
                                      cv2.VideoWriter_fourcc(*'DIVX'), 
                                      fps, (w,h))
            window_height = h//16
            window_top, window_bottom = h//2, h//2 + window_height
            window_width = int(w*0.4)
            window_left, window_right = w//2-window_width//2, w//2+window_width//2
            window = ROI((window_left, window_top, window_right, window_bottom))
            roi_resize = (int((window_right-window_left)*scale),
                          int((window_bottom-window_top)*scale))
        
            while cap.isOpened():
                ret, frame = cap.read()
                if ret is False:
                    print('ret false',file=open('log.log','a'))
                    break
                else:
                    time = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                    timestr = f'Time: {time:.1f} s'
                    if end_time is None or time <= end_time:
                        if detect:
#                            print(window.left//2+window.right//2,
#                                  window.top//2+window.bottom//2,
#                                  file=open('window_track.txt','a'))
                            roi_left = ROI((window.left,
                                            window.top,
                                            window.left+int(window.size()[0]*0.4),
                                            window.bottom))
                            roi_right = ROI((window.left+int(window.size()[0]*0.6),
                                            window.top,
                                            window.right,
                                            window.bottom))
                            for roi in (roi_left, roi_right):
                                roi.set_source(frame)
                                roi.resize(roi_resize)
                            roi = np.concatenate((roi_left.img, roi_right.img), axis=1)
                            fraction, proba = self.get_fraction(roi)
                            if len(self.stack) < self.stack_size:
                                self.stack.append((fraction,proba))
                            else:
                                self.stack = self.stack[1:] + [(fraction,proba)]
                            fraction, proba = self.stack_vote()
                            fractions.append(fraction)
                            probs.append(proba)
                            times.append(time)
        
                            if to_print:
                                print( f'Time: {time:.2f} s\tFraction: {fraction}\tProbability: {proba:.3f}' )
                        if to_show or to_write:
                            display = frame
                            if not detect:
                                clr = (255,255,255)
                                cv2.putText(display, 'Detection OFF',
                                            (window.left, window.top-4),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            1, clr, 2)
                            else:
                                text_bottom = window.bottom-4
                                text_width = (window.size()[0]-4)//self.n_classes
                                try:
                                    idx = classes.index(fraction)
                                    clr = colors[idx]
                                    text_left = window.left + 4 + idx*text_width
                                except Exception:
                                    clr = (0,0,0)
                                    text_left = window.left//2 + window.right//2 - text_width//2                            
                                cv2.putText(display, 'Detection ON',
                                            (window.left, window.top-4),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            1, clr, 2)
                                cv2.putText(display, fraction,
                                            (text_left, text_bottom),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            2, clr, 4)
                            cv2.rectangle(display, (window.left, window.top),
                                          (window.right, window.bottom),
                                          clr, 2 )
                            cv2.putText(display, timestr,
                                        (window.left//2 + window.right//2, 
                                         window.top-4),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1, clr, 2)
                            if to_write:
                                out.write(display)
                            else:
                                cv2.putText(display, 'Press Esc to quit, Space to start/stop detection',
                                            (4, h-8),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            2, (255,255,255), 2)
                            if to_show:
                                cv2.imshow('Classification result', display)
                                k = cv2.waitKey(delay) & 0xFF
                                if k == 27:
                                    break
                                elif k == 32:
                                    detect = not detect
                                elif k == ord('a') and window.left>step: #left arrow
                                    window.left -= step
                                    window.right -= step
                                elif k == ord('d') and window.right<w-step: #right arrow
                                    window.left += step
                                    window.right += step
                                elif k == ord('w') and window.top>step: #up arrow
                                    window.top -= step
                                    window.bottom -= step
                                elif k == ord('s') and window.bottom<h-step: #down arrow
                                    window.top += step
                                    window.bottom += step

#                                elif k == 81 and window.left>step: #left arrow
#                                    window.left -= step
#                                    window.right -= step
#                                elif k == 83 and window.right<w-step: #right arrow
#                                    window.left += step
#                                    window.right += step
#                                elif k == 82 and window.top>step: #up arrow
#                                    window.top -= step
#                                    window.bottom -= step
#                                elif k == 84 and window.bottom<h-step: #down arrow
#                                    window.top += step
#                                    window.bottom += step
                    else:
                        print(f'time is out',file=open('log.log','a'))
                        break
        except Exception as e:
            print(str(e), file=open('log.log','a'))
        if cap.isOpened():
            print('releasing cap',file=open('log.log','a'))
            cap.release()
        if to_write and out.isOpened():
            print('releasing out',file=open('log.log','a'))
            out.release()            
        if to_show:
            print('destroying windows',file=open('log.log','a'))
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        return times, fractions, probs
       
