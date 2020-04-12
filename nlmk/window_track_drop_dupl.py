#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
w = pd.read_csv('window_track.txt', header=None, sep=' ')
w.drop_duplicates(inplace=True)
w.to_csv('window_track.txt', header=None, index=False, sep=' ')

