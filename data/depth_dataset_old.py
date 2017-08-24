import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import PIL
from pdb import set_trace as st
import numpy as np


class DepthDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        self.dir_C = os.path.join(opt.dataroot, opt.phase + 'C')    # for depth dataset

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)
        self.C_paths = make_dataset(self.dir_C)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.C_paths = sorted(self.C_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.C_size = len(self.C_paths)
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size]
        C_path = self.C_paths[index % self.C_size]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        C_img = Image.open(C_path).convert('RGB')

        # concatenate A and C to get consistent transform
        A_array = np.array(A_img)
        C_array = np.array(C_img)
        AC_array = np.concatenate((A_array,C_array), axis=2)
        AC_img = Image.fromarray(AC_array)

        AC_img = self.transform(AC_img)
        B_img = self.transform(B_img)

        # split AC into A and C
        AC_array = np.array(AC_img)
        A_array = AC_array[:,:,:3]
        C_array = C_array[:,:,3:]
        A_img = Image.fromarray(A_array)
        C_img = Image.fromarray(C_array)

        return {'A': A_img, 'B': B_img, 'C': C_img,
                'A_paths': A_path, 'B_paths': B_path, 'C_paths': C_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'DepthDataset'
