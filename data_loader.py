#!/usr/bin/env python
# coding: utf-8

import os, re
import numpy as np

#Function to parse our files
def parse_text(file, dir):
    with open(dir + file, 'rt') as fd:
        data=[]
        line = fd.readline()
        nline = re.findall(r"[-+]?\d*\.\d+|\d+", line)
        data.append(nline)
        while line:
            line=fd.readline()
            nline = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            data.append(nline)
    return data

#Function to load the data
def load_data():
    #Open directory and files
    os.chdir(os.getcwd())
    base_dir = 'Raman_Data/'
    als_dir = base_dir + 'ALS/'
    ctrl_dir = base_dir + 'CTRL/'
    
    all_files_als = os.listdir(als_dir)
    all_files_ctrl = os.listdir(ctrl_dir)

    all_files_als.sort(key=lambda f: int(re.sub('\D', '', f)))
    all_files_ctrl.sort(key=lambda f: int(re.sub('\D', '', f)))

    #Start extraction
    X=[] #actual y of spectra
    Y=[] # 1 -> als; 0 -> ctrl
    coord=[] #actual x of spectra

    sep=[60,78,114,150,194,210,225,241,255,280,299,313,323,333,343,353,363,373,383,393] #Il manque le 227
    groups=[] #for GROUP K FOLD
    group=0
    index=1
    for f in all_files_als:
      data=[]
      datab=[]
      for e in parse_text(f, als_dir):
        if len(e) > 0:
            datab.append(float(e[0]))
            data.append(float(e[1]))
      coord.append(datab)
      X.append(data)
      Y.append(1)
      groups.append(group)
      if index in sep:
        group+=1
      index+=1
    sep=[33,76,91,138,149,158,168,178,188,198]
    index=1
    for f in all_files_ctrl:
      data=[]
      datab=[]
      for e in parse_text(f, ctrl_dir):
        if len(e) > 0:
          datab.append(float(e[0]))
          data.append(float(e[1]))
      coord.append(datab)
      X.append(data)
      Y.append(0)
      groups.append(group)
      if index in sep:
        group+=1
      index+=1
    
    #Transform into np.array
    X=np.array(X)
    Y=np.array(Y)
    groups=np.array(groups)
    
    #Remove the negative values from spectra
    for i in range(len(X)):
        for j in range (len(X[i])):
            if(X[i][j] < 0):
                X[i][j] = 0
    
    #Return X, Y, groups and coord
    return X, Y, groups, coord

