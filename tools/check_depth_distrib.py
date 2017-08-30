"""
Check the distribution of depth images
"""

from PIL import Image
import glob
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys

#root = '/home/xyang/UTS/Data/Haze/D-HAZY/NYU_GT/'
#name = 'disentangled_shuffle_resnet_9blocks_sigmoid_A100_TV1_lr0.0002';
name = sys.argv[1]

root = '/home/xyang/UTS/Data/Haze/D-HAZY/NYU/results/'+name+'/test_latest/images/'

reverse = True

filenames = glob.glob(root+'*Estimate_depth*')
images = []
for name in filenames:
    img = Image.open(name)
    if reverse:
        images.append(255. - np.array(img))
    else:
        images.append(np.array(img))

images = np.vstack(images)

hist, _ = np.histogram(images, 20)
hist = hist / float(np.sum(hist))
print hist
l1, = plt.plot(range(0,255,int(np.ceil(255./20))), hist, 'r', label='Depth intensity')

a = np.linspace(0,1,20)
a = np.exp(-a)
#a = 1-a
a /= np.sum(a)
l2, = plt.plot(range(0,255,int(np.ceil(255./20))), a, 'g', label='exp(-x)')
plt.legend(handles=[l1, l2])
#plt.show()
plt.savefig('depth_distrib.png')
