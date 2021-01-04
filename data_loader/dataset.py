import os
from glob import glob
from matplotlib.image import imread
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset

from utils.imtools import for_fft
from utils import comfft as cf

'''For the test of Lai_Real Dataset'''
class Test_Dataset(Dataset):
    def __init__(self, test_sp_dir, test_bl_dir, test_ker_dir):

        self.bl_dir = test_bl_dir
        self.sp_dir = test_sp_dir
        self.ker_dir = test_ker_dir
        self.sp_file = sorted(glob(self.sp_dir + '*.png'))

        ker_mat = loadmat(test_ker_dir)
        ker_mat = ker_mat['kernels']
        self.get_ker = lambda idx: ker_mat[0, idx]
        self.ker_num = len(ker_mat[0])


    def  __len__(self):
        img_num = len(self.sp_file) * self.ker_num
        return img_num

    def __getitem__(self, item):
        i = item // self.ker_num
        j = item % self.ker_num
        sp = imread(os.path.join(self.sp_dir, 'im_%d.png'%(i+1)))
        bl = imread(os.path.join(self.bl_dir, 'im_%d_ker_%d.png'%(i+1,j+1)))

        ker = self.get_ker(j)
        ker_pad = np.full([50, 50], np.nan)
        ker_pad[:ker.shape[0], :ker.shape[1]] = ker

        ker_mat = torch.FloatTensor(for_fft(ker, shape=np.shape(sp)))
        Fker = cf.fft(ker_mat)

        sp = torch.from_numpy(sp).unsqueeze(0)
        bl = torch.from_numpy(bl).unsqueeze(0)
        dic = {'bl': bl, 'sp': sp, 'Fker':Fker, 'ker': ker_pad.copy()}
        return dic


