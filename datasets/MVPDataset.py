import torch
import numpy as np
import torch.utils.data as data
import h5py
import os
import transforms3d
import open3d as o3d
import random
import math
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from .io import IO
import random
import os
import json
from .build import DATASETS
from utils.logger import *

cat_name = {0:'airplane',
            1:'cabinet', 
            2:'car', 
            3:'chair', 
            4:'lamp', 
            5:'sofa', 
            6:'table', 
            7:'watercraft', 
            8:'bed', 
            9:'bench', 
            10:'bookshelf', 
            11:'bus', 
            12:'guitar', 
            13:'motorbike', 
            14:'pistol', 
            15:'skateboard'}

cat_taxonomy_id = {0:'02691156',
            1:'02933112', 
            2:'02958343', 
            3:'03001627', 
            4:'03636649', 
            5:'04256520', 
            6:'04379243', 
            7:'04530566', 
            8:'02818832', 
            9:'02828884', 
            10:'02871439', 
            11:'02924116', 
            12:'03467517', 
            13:'03790512', 
            14:'03948459', 
            15:'04225987'}
@DATASETS.register_module()
class MVP(data.Dataset):
    def __init__(self, config):
        self.aug = config.AUG
        self.subset = config.subset
        self.npoints = config.N_POINTS
        self.novel_input_only = False
        self.novel_input = True
        self.input_path = config.PARTIAL_POINTS_PATH % self.subset
        self.gt_path = config.COMPLETE_POINTS_PATH % (self.subset, self.npoints)
        
        input_file = h5py.File(self.input_path, 'r')
        self.input_data = np.array((input_file['incomplete_pcds'][()]))
        self.labels = np.array((input_file['labels'][()]))
        self.novel_input_data = np.array((input_file['novel_incomplete_pcds'][()]))
        self.novel_labels = np.array((input_file['novel_labels'][()]))
        input_file.close()

        gt_file = h5py.File(self.gt_path, 'r')
        self.gt_data = np.array((gt_file['complete_pcds'][()]))
        self.novel_gt_data = np.array((gt_file['novel_complete_pcds'][()]))
        gt_file.close()

        if self.novel_input_only:
            self.input_data = self.novel_input_data
            self.gt_data = self.novel_gt_data
            self.labels = self.novel_labels
        elif self.novel_input:
            self.input_data = np.concatenate((self.input_data, self.novel_input_data), axis=0)
            self.gt_data = np.concatenate((self.gt_data, self.novel_gt_data), axis=0)
            self.labels = np.concatenate((self.labels, self.novel_labels), axis=0)

        print('######## MVP dataset ########')
        print(self.input_data.shape)
        print(self.gt_data.shape)
        print(self.labels.shape)
        print('#############################')
        self.len = self.input_data.shape[0]

        self.pc_augm_scale = 1.5
        self.pc_augm_rot = 1 # 1
        self.pc_augm_mirror_prob = 0.5 # 0.5
        self.pc_augm_jitter = 0

    def augment_cloud(self, Ps):
        """" Augmentation on XYZ and jittering of everything """
        M = transforms3d.zooms.zfdir2mat(1)
        if self.pc_augm_scale > 1:
#            print('scale')
            s = random.uniform(1/self.pc_augm_scale, self.pc_augm_scale)
            M = np.dot(transforms3d.zooms.zfdir2mat(s), M)
        if self.pc_augm_rot:
#            print('rot')
            if random.random() < self.pc_augm_rot/2:
                angle = random.uniform(0, 2*math.pi)
                M = np.dot(transforms3d.axangles.axangle2mat([0,1,0], angle), M) # y=upright assumption
            if random.random() < self.pc_augm_rot/2:
                angle = random.uniform(0, 2*math.pi)
                M = np.dot(transforms3d.axangles.axangle2mat([1,0,0], angle), M) # y=upright assumption
            if random.random() < self.pc_augm_rot/2:
                angle = random.uniform(0, 2*math.pi)
                M = np.dot(transforms3d.axangles.axangle2mat([0,0,1], angle), M) # y=upright assumption
        if self.pc_augm_mirror_prob > 0: # mirroring x&z, not y
#            print('mirror')
            if random.random() < self.pc_augm_mirror_prob/2:
                M = np.dot(transforms3d.zooms.zfdir2mat(-1, [1,0,0]), M)
            if random.random() < self.pc_augm_mirror_prob/2:
                M = np.dot(transforms3d.zooms.zfdir2mat(-1, [0,0,1]), M)
        result = []
        for P in Ps:
            P[:,:3] = np.dot(P[:,:3], M.T)

            if self.pc_augm_jitter:
                sigma, clip= 0.01, 0.05 # https://github.com/charlesq34/pointnet/blob/master/provider.py#L74
                P = P + np.clip(sigma * np.random.randn(*P.shape), -1*clip, clip).astype(np.float32)
            result.append(P)
        return result[0], result[1]


    def __len__(self):
        return self.len

    def __getitem__(self, index):
        partial = torch.from_numpy((self.input_data[index]))
        complete = torch.from_numpy((self.gt_data[index // 26]))
        if self.subset == 'train' and self.aug:
            partial, complete = self.augment_cloud([partial,complete])
        taxonomy_id = cat_taxonomy_id[int(self.labels[index])]
        model_id = cat_taxonomy_id[int(self.labels[index])]
        return taxonomy_id, model_id, (partial, complete)