import os
import pydicom
import numpy as np
import gzip
from skimage.transform import resize
import imageio
import pickle

goalpath=os.path.join(os.getcwd(), 'main_dir')
target_dir='train_data_RWMAv2temp'

if not os.path.isdir(target_dir):
    os.makedirs(target_dir)
save_path=os.path.join(os.getcwd(), target_dir)

with open('all_labels.pickle', 'rb') as handle:
    all_labels=pickle.load(handle)

def main(all_labels):
    anlist=all_labels.keys()
    for an in anlist:
        print(an, ' is being processed.', '{}/{}'.format(anlist.index(an), len(anlist)))
        labels2savefile(all_labels[an], an)



def labels2savefile(labels, an):
    an_path=os.path.join(goalpath, an)
    name_list=['LAX', 'SAX', '4ch', '2ch']
    for seq, i in enumerate(labels.argmax(axis=0)[:4]):
        file_path=os.path.join(an_path, files[i])
        tar_array=savingfile(file_path, a=768, b=1024)
        save_file_path=os.path.join(save_path, an, name_list[seq]+'.npy.gz')
        img_file_path=os.path.join(save_path, an, name_list[seq]+'.jpg')
        with gzip.GzipFile(save_file_path, "w") as gf:
            np.save(gf, tar_array)
        imageio.imwrite(img_file_path, tar_array[0][5])
        
        
        

def get_array(path):
    ds=pydicom.read_file(path)
    dt=ds.pixel_array
    return dt

def regrid_vid(tar_array, a=600, b=800):
    for i, img in enumerate(tar_array):
        img=regrid(img, a=a, b=b)
        if i==0: tar_array_sub=img
        else : tar_array_sub=np.concatenate((tar_array_sub, img))
    return tar_array_sub

def regrid(img, a=600, b=800):
    img_resized=resize(img, (a,b))
    img_resized=np.expand_dims(img_resized, axis=0)
    return img_resized

def pad2fifty(tar_dcm):
    add_len=50-len(tar_dcm)
    add_array=np.zeros(shape=(add_len, tar_dcm.shape[1], tar_dcm.shape[2], tar_dcm.shape[3]))
    tar_dcm=np.concatenate((tar_dcm, add_array))
    return tar_dcm

def preparefile(file_path, a=600, b=800):
    #print(tar_array.shape)
    tar_array=get_array(file_path)
    if len(tar_array.shape) < 4: raise NotImplementedError
    tar_array=tar_array/255
    if tar_array.shape[1:4] != (a, b, 3):
        tar_array=regrid_vid(tar_array, a, b)
    if len(tar_array)<50:
        tar_array=pad2fifty(tar_array)   
    tar_array=tar_array[:50]
    tar_array=np.expand_dims(tar_array, axis=0)
    return tar_array

def savingfile(file_path, a=600, b=800):
    tar_array=get_array(file_path)/255
    #print(tar_array.shape)
    if tar_array.shape[1:4] != (a, b, 3):
        tar_array=regrid_vid(tar_array, a, b)
    if len(tar_array)<50:
        tar_array=pad2fifty(tar_array)   
    tar_array=tar_array[:50]
    tar_array=np.expand_dims(tar_array, axis=0)
    tar_array=tar_array*255
    tar_array=tar_array.astype('uint8')
    return tar_array


if __name__=='__main__':
    main(all_labels)