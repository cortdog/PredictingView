# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 03:59:59 2018

@author: Adrian
"""

import os
import numpy as np
import gzip
from skimage.transform import resize
import multiprocessing as mp
import sys


train_path=os.path.join(os.getcwd(), 'train_data')

total_numbers=len(os.listdir(train_path))

def regrid(img, x=600, y=800):
    img_resized=resize(img, (x,y))
    return img_resized


def resize_all(i):

    print('{}/{}'.format(i, total_numbers))
    file='{}.npy.gz'
    
    with gzip.GzipFile(os.path.join(train_path, file.format(i)), "r") as gf:
        tar_array=np.load(gf)
    boo = tar_array.shape==(50,600,800,3)
    
    if len(tar_array.shape) < 4:
        print('No RGB')
        tar_array=np.expand_dims(tar_array, axis=-1)
        tar_array=tar_array.repeat(3, axis=-1)
        
    
    if tar_array.dtype=='float64':
        print('down sizing')
        tar_array=tar_array*255
        tar_array=tar_array.astype('uint8')
        with gzip.GzipFile(os.path.join(train_path, file.format(i)), "w") as gf:
            np.save(gf, tar_array)             
    
    if not boo:
        print('rescaling')
        for j, img in enumerate(tar_array):
            img=regrid(img)
            img=np.expand_dims(img, axis=0)
            if j==0: tar_array_sub=img
            else: tar_array_sub=np.concatenate((tar_array_sub, img))
        tar_array=tar_array_sub*255
        tar_array=tar_array.astype('uint8')
        with gzip.GzipFile(os.path.join(train_path, file.format(i)), "w") as gf:
            np.save(gf, tar_array) 
    sys.stdout.flush()
  
def multi_process(x):
    pool=mp.Pool(processes=3)
    r=pool.map_async(resize_all, range(x, total_numbers))
    r.wait()

    
    
#if __name__=='__main__':
#    multi_process(3030)
    