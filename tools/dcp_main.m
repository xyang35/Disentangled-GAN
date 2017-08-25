import os
import argparse
from functools import partial
import pdb
import glob

from PIL import Image

from util import get_filenames
from dehaze import dehaze

SP_IDX = (5, 6, 8, 12)  # for testing parameters
SP_PARAMS = ({'tmin': 0.2, 'Amax': 170, 'w': 15, 'r': 40},
             {'tmin': 0.5, 'Amax': 190, 'w': 15, 'r': 40},
             {'tmin': 0.5, 'Amax': 220, 'w': 15, 'r': 40})


def generate_results(src, dest, generator):
    print 'processing', src + '...'
    im = Image.open(src)
    dark, rawt, refinedt, rawrad, rerad = generator(im)
    dark.save(dest % 'dark')
    rawt.save(dest % 'rawt')
    refinedt.save(dest % 'refinedt')
    rawrad.save(dest % 'radiance-rawt')
    rerad.save(dest % 'radiance-refinedt')
    print 'saved', dest


def main():
#    filenames = get_filenames()
#    parser = argparse.ArgumentParser()
#    parser.add_argument("-i", "--input", type=str,
#                        help="index for single input image")
#    parser.add_argument("-t", "--tmin", type=float, default=0.2,
#                        help="minimum transmission rate")
#    parser.add_argument("-A", "--Amax", type=int, default=220,
#                        help="maximum atmosphere light")
#    parser.add_argument("-w", "--window", type=int, default=15,
#                        help="window size of dark channel")
#    parser.add_argument("-r", "--radius", type=int, default=40,
#                        help="radius of guided filter")
#
#    args = parser.parse_args()

    tmin = 0.1
    Amax = 220
    window = 15
    radius = 40


    #root = '/home/xyang/Downloads/GAN/disentangled_resnet_9blocks_sigmoid_A100_TV0.00001/disentangled_resnet_9blocks_sigmoid_A100_TV0.00001/test_latest/images/';
    name = 'disentangled_resnet_9blocks_sigmoid_A100_TV1'
    
    root = '/home/xyang/UTS/Data/Haze/D-HAZY/NYU/results/'+name+'/test_latest/images/'
    output_root = '/home/xyang/Downloads/GAN/DCP/'+name+'/';
    os.mkdir(output_root)

    img_names = glob.glob(root+'*_Hazy_Hazy.png')
    for i, name in enumerate(img_names):
        src = name
        name = os.path.basename(name)
        dest = output_root+name.replace('_Hazy.png', '_dcp_%s.png')

        print name

        generate_results(src, dest, partial(dehaze, tmin=tmin, Amax=Amax,
                                           w=window, r=radius))




#    if args.input is not None:
#        src, dest = filenames[args.input]
#        src = args.input
#        dest = 'result_%s.png'
#        dest = dest.replace("%s",
#                     "%s-%d-%d-%d-%d" % ("%s", args.tmin * 100, args.Amax,
#                                           args.window, args.radius))
#        generate_results(src, dest, partial(dehaze, tmin=args.tmin, Amax=args.Amax,
#                                           w=args.window, r=args.radius, omega=1))
#    else:
#        for idx in SP_IDX:
#            src, dest = filenames[idx]
#            for param in SP_PARAMS:
#                newdest = dest.replace("%s",
#                     "%s-%d-%d-%d-%d" % ("%s", param['tmin'] * 100,
#                                         param['Amax'], param['w'],
#                                         param['r']))
#                generate_results(src, newdest, partial(dehaze, **param))
#
#        for src, dest in filenames:
#            generate_results(src, dest, dehaze)

if __name__ == '__main__':
    main()
