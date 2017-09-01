import PIL
from PIL import Image
from multiprocessing import Pool
import os
import glob
import sys
import argparse
import pdb
import numpy as np

def reverse_matting(I, t, A=1, t0=0.01):
    """
    I -- image with range [0,1]
    t -- image with range [0,1]
    """

    t_clamp = np.clip(t, t0, 1)
    J = (I - A) / np.expand_dims(t_clamp, axis=2) + A

    return np.clip(J, 0, 1)

def func(img_name):
    t_name = img_name.replace('_Hazy.png', '_Estimate_depth_'+args.suffix+'.png')
#    t_name = img_name.replace('_Hazy.png', '_real_depth.png')

    img = Image.open(img_name).convert('RGB')
    I = np.array(img, dtype=float)
    I /= 255.

    t_img = Image.open(t_name).convert('L')
    t = np.array(t_img, dtype=float)
    t /= 255.

    J = reverse_matting(I, t)
    J *= 255

    J_img = Image.fromarray(J.astype('uint8'))
    output_name = t_name.replace('Estimate_depth', 'Haze-free-depth')
#    output_name = t_name.replace('real_depth', 'Haze-free-real-depth')
    print "Saving to ", output_name
    J_img.save(output_name)

def main(args):

    root = '/home/xyang/UTS/Data/Haze/D-HAZY/NYU/results/'+args.name+'/test_latest/images/'
    img_names = glob.glob(root+'*_Hazy_Hazy.png')    # original hazy images

    if args.pool_size > 1:
        p = Pool(args.pool_size)
        p.map(func, img_names)
    else:
        for img_name in img_names:
            func(img_name)
    
if  __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='configuration name')
    parser.add_argument('--suffix', type=str, default='sigma', help='configuration name')
    parser.add_argument('--pool_size', type=int, default=1, help='pool size')

    args = parser.parse_args()
    
    main(args)
