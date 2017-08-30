from scipy.io import loadmat
import numpy as np
import sys
import pdb

#name = 'disentangled_shuffle_resnet_9blocks_sigmoid_A100_TV1_lr0.0002';
name = sys.argv[1]

root = '/home/xyang/UTS/Data/Haze/D-HAZY/NYU/results/'+name+'/test_latest/images/'

suffix = '_dcp_radiance-refinedt'; folder = '/home/xyang/Downloads/GAN/demo/app/static/DCP/' 
#suffix = '_DehazeNet'; folder = 'DehazeNet/'+name+'/'
#suffix = '_Haze-free'; folder = root

eval1 = loadmat(root+'evaluation.mat')
eval2 = loadmat(folder+'evaluation.mat')

for i in range(len(eval1['img_names'])):
    if not eval1['img_names']['name'][i] == eval2['img_names']['name'][i]:
        print i, "doesn't match"
        raise

psnr_diff = eval1['peaksnr_all'] - eval2['peaksnr_all']
ssim_diff = eval1['ssim_all'] - eval2['ssim_all']

psnr_idx = np.argsort(psnr_diff, axis=0)
ssim_idx = np.argsort(ssim_diff, axis=0)

psnr_bad_names = eval1['img_names']['name'][psnr_idx[:5]]
psnr_good_names = eval1['img_names']['name'][psnr_idx[-5:]]
ssim_bad_names = eval1['img_names']['name'][ssim_idx[:5]]
ssim_good_names = eval1['img_names']['name'][ssim_idx[-5:]]

# ours only
our_psnr = eval1['peaksnr_all']
our_ssim = eval1['ssim_all']
our_psnr_idx = np.argsort(our_psnr, axis=0)
our_ssim_idx = np.argsort(our_ssim, axis=0)

our_psnr_bad_names = eval1['img_names']['name'][our_psnr_idx[:5]]
our_psnr_good_names = eval1['img_names']['name'][our_psnr_idx[-5:]]
our_ssim_bad_names = eval1['img_names']['name'][our_ssim_idx[:5]]
our_ssim_good_names = eval1['img_names']['name'][our_ssim_idx[-5:]]

from visualize import HTML

web_name = 'Check_evaluation_'+name
webpage = HTML(web_name)

# for output
root = '/static/'+name+'/images/'
folder = '/static/DCP/'

webpage.add_header('PSNR: our worst results')
for i, img_name in enumerate(our_psnr_bad_names):
    print img_name
    img_path = []
    n = img_name[0][0][0]
    img_path.append(root + n.replace('_real_B.png', '_Hazy.png'))
    img_path.append(root + n.replace('_real_B.png', '_Haze-free.png'))
    img_path.append(folder+n.replace('_real_B.png', suffix+'.png'))
    img_path.append(root + n)

    val = []
    val.append('Hazy')
    val.append(str(eval1['peaksnr_all'][our_psnr_idx[i]][0]))
    val.append(str(eval2['peaksnr_all'][our_psnr_idx[i]][0]))
    val.append('real_B')
    webpage.add_images(img_path, val, 256)

webpage.add_header('PSNR: our best results')
for i, img_name in enumerate(our_psnr_good_names):
    print img_name
    img_path = []
    n = img_name[0][0][0]
    img_path.append(root + n.replace('_real_B.png', '_Hazy.png'))
    img_path.append(root + n.replace('_real_B.png', '_Haze-free.png'))
    img_path.append(folder+n.replace('_real_B.png', suffix+'.png'))
    img_path.append(root + n)

    val = []
    val.append('Hazy')
    val.append(str(eval1['peaksnr_all'][our_psnr_idx[-(i+1)]]))
    val.append(str(eval2['peaksnr_all'][our_psnr_idx[-(i+1)]]))
    val.append('real_B')
    webpage.add_images(img_path, val, 256)

webpage.add_header('SSIM: our worst results')
for i, img_name in enumerate(our_ssim_bad_names):
    print img_name
    img_path = []
    n = img_name[0][0][0]
    img_path.append(root + n.replace('_real_B.png', '_Hazy.png'))
    img_path.append(root + n.replace('_real_B.png', '_Haze-free.png'))
    img_path.append(folder+n.replace('_real_B.png', suffix+'.png'))
    img_path.append(root + n)

    val = []
    val.append('Hazy')
    val.append(str(eval1['ssim_all'][our_ssim_idx[i]]))
    val.append(str(eval2['ssim_all'][our_ssim_idx[i]]))
    val.append('real_B')
    webpage.add_images(img_path, val, 256)

webpage.add_header('SSIM: our best results')
for i, img_name in enumerate(our_ssim_good_names):
    print img_name
    img_path = []
    n = img_name[0][0][0]
    img_path.append(root + n.replace('_real_B.png', '_Hazy.png'))
    img_path.append(root + n.replace('_real_B.png', '_Haze-free.png'))
    img_path.append(folder+n.replace('_real_B.png', suffix+'.png'))
    img_path.append(root + n)

    val = []
    val.append('Hazy')
    val.append(str(eval1['ssim_all'][our_ssim_idx[-(i+1)]]))
    val.append(str(eval2['ssim_all'][our_ssim_idx[-(i+1)]]))
    val.append('real_B')
    webpage.add_images(img_path, val, 256)

webpage.add_header('Compare our results with DCP')

webpage.add_header('PSNR: our %s worse than %s' % (name,suffix))
for i, img_name in enumerate(psnr_bad_names):
    print img_name
    img_path = []
    n = img_name[0][0][0]
    img_path.append(root + n.replace('_real_B.png', '_Hazy.png'))
    img_path.append(root + n.replace('_real_B.png', '_Haze-free.png'))
    img_path.append(folder+n.replace('_real_B.png', suffix+'.png'))
    img_path.append(root + n)

    val = []
    val.append('Hazy')
    val.append(str(eval1['peaksnr_all'][psnr_idx[i]][0]))
    val.append(str(eval2['peaksnr_all'][psnr_idx[i]][0]))
    val.append('real_B')
    webpage.add_images(img_path, val, 256)

webpage.add_header('PSNR: our %s better than %s' % (name,suffix))
for i, img_name in enumerate(psnr_good_names):
    print img_name
    img_path = []
    n = img_name[0][0][0]
    img_path.append(root + n.replace('_real_B.png', '_Hazy.png'))
    img_path.append(root + n.replace('_real_B.png', '_Haze-free.png'))
    img_path.append(folder+n.replace('_real_B.png', suffix+'.png'))
    img_path.append(root + n)

    val = []
    val.append('Hazy')
    val.append(str(eval1['peaksnr_all'][psnr_idx[-(i+1)]]))
    val.append(str(eval2['peaksnr_all'][psnr_idx[-(i+1)]]))
    val.append('real_B')
    webpage.add_images(img_path, val, 256)

webpage.add_header('SSIM: our %s worse than %s' % (name,suffix))
for i, img_name in enumerate(ssim_bad_names):
    print img_name
    img_path = []
    n = img_name[0][0][0]
    img_path.append(root + n.replace('_real_B.png', '_Hazy.png'))
    img_path.append(root + n.replace('_real_B.png', '_Haze-free.png'))
    img_path.append(folder+n.replace('_real_B.png', suffix+'.png'))
    img_path.append(root + n)

    val = []
    val.append('Hazy')
    val.append(str(eval1['ssim_all'][ssim_idx[i]]))
    val.append(str(eval2['ssim_all'][ssim_idx[i]]))
    val.append('real_B')
    webpage.add_images(img_path, val, 256)

webpage.add_header('SSIM: our %s better than %s' % (name,suffix))
for i, img_name in enumerate(ssim_good_names):
    print img_name
    img_path = []
    n = img_name[0][0][0]
    img_path.append(root + n.replace('_real_B.png', '_Hazy.png'))
    img_path.append(root + n.replace('_real_B.png', '_Haze-free.png'))
    img_path.append(folder+n.replace('_real_B.png', suffix+'.png'))
    img_path.append(root + n)

    val = []
    val.append('Hazy')
    val.append(str(eval1['ssim_all'][ssim_idx[-(i+1)]]))
    val.append(str(eval2['ssim_all'][ssim_idx[-(i+1)]]))
    val.append('real_B')
    webpage.add_images(img_path, val, 256)

webpage.save()
