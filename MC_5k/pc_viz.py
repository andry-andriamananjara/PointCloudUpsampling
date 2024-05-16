import os, sys
sys.path.append("../")

import argparse
import utils.data_util as utils
import h5py
import cv2
import numpy as np
from utils.pc_util import draw_point_cloud
from Common import point_operation

def plot_save_pcd(pcd, file, exp_name):

    image_save_dir=os.path.join("viz_sample",exp_name)

    if os.path.exists(image_save_dir)==False:
        os.makedirs(image_save_dir)

    file_name=file.split("/")[-1].split('.')[0]
    img = draw_point_cloud(pcd, zrot=90 / 180.0 * np.pi, xrot=90 / 180.0 * np.pi, yrot=0 / 180.0 * np.pi,
                           diameter=4)
    img=(img*255).astype(np.uint8)
    image_save_path=os.path.join(image_save_dir,file_name+".png")
    cv2.imwrite(image_save_path,img)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--file_type', type=str, required=True, help='punet or pu1k or xyz')
    parser.add_argument('--path', type=str, required=True, help='path file')
    parser.add_argument('--uniform',action='store_true', default=False)
    args = parser.parse_args()
    
    file_type = args.file_type
    path      = args.path
    uniform   = args.uniform
    exp_name  = "viz_test"
    
    if file_type == 'punet':
        gt_h5_file = h5py.File(path)
        if uniform:
            #path = "Mydataset/PUNET/uniform_256_1024_test.h5"
            #path = "Mydataset/PUNET/non_uniform_256_1024_train.h5"
            
            print(gt_h5_file.keys())
            gt     = gt_h5_file['gt'][:]
            input  = gt_h5_file['input'][:]
        else:
            print(gt_h5_file.keys())
            gt     = gt_h5_file['gt'][:]
            input  = gt_h5_file['input'][:]

    elif file_type == 'pu1k':
        gt_h5_file = h5py.File(path)
        if uniform:
            #path = "Mydataset/PUNET/uniform_256_1024_test.h5"
            #path = "Mydataset/PUNET/non_uniform_256_1024_train.h5"
            print(gt_h5_file.keys())
            gt     = gt_h5_file['poisson_1024'][:]
            input  = gt_h5_file['poisson_256'][:]
        else:
            print(gt_h5_file.keys())
            gt     = gt_h5_file['poisson_1024'][:]
            input  = gt_h5_file['poisson_1024'][:]
    
    else:
        pcd = np.loadtxt(path)
        plot_save_pcd(pcd, path, exp_name)
    
    ## Plot
    for num_idx in range(5):
        print(gt.shape, input.shape)
        print(input[0,:,:])
        print(gt[0,:,:])
        pcd     = gt[num_idx,:,:]
        plot_save_pcd(pcd, path.replace('.',str(num_idx)+'_gt.'), exp_name)

        pcd     = input[num_idx,:,:]
        if not uniform:
            idx = point_operation.nonuniform_sampling(1024, sample_num=256)
            pcd = gt[num_idx,:,:]
            pcd = pcd[idx,:]
        plot_save_pcd(pcd, path.replace('.',str(num_idx)+'_input.'), exp_name)
        
        
        