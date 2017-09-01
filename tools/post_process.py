import PIL
from PIL import Image
from multiprocessing import Pool
import os
import glob
import sys
import argparse
import pdb
import numpy as np


def func_sigma(img_name):
    from sigma_filter import SigmaFilter

    img = Image.open(img_name).convert('L')
    I = np.array(img, dtype=float)
    I /= 255.

    sigma_filter = SigmaFilter()
    result = sigma_filter(I)
    result *= 255

    result_img = Image.fromarray(result.astype('uint8'))
    output_name = img_name.replace('Estimate_depth', 'Estimate_depth_sigma')
    print "Saving to ", output_name
    result_img.save(output_name)
    
def func_strech(img_name):
    img = Image.open(img_name).convert('L')
    I = np.array(img, dtype=float)

    new_I = (I - I.min()) / (I.max() - I.min()) * 255.
    result_img = Image.fromarray(new_I.astype('uint8'))
    output_name = img_name.replace('Estimate_depth', 'Estimate_depth_strech')
    print "Saving to ", output_name
    result_img.save(output_name)


def main(args):

    root = '/home/xyang/UTS/Data/Haze/D-HAZY/NYU/results/'+args.name+'/test_latest/images/'
    img_names = glob.glob(root+'*_Estimate_depth.png')    # post-process on transmission images

    if args.pool_size > 1:
        p = Pool(args.pool_size)
        if args.method == 'sigma':
            p.map(func_sigma, img_names)
        elif args.method == 'strech':
            # histogram streching for increasing contrast: https://www.tutorialspoint.com/dip/histogram_stretching.htm
            p.map(func_strech, img_names)

    else:
        for img_name in img_names:
            func_sigma(img_name)


if  __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='configuration name')
    parser.add_argument('--method', type=str, default='sigma', help='configuration name')
    parser.add_argument('--pool_size', type=int, default=1, help='pool size')

    args = parser.parse_args()
    
    main(args)
