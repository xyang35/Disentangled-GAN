import dominate
from dominate.tags import *
import os
import numpy as np
import ntpath
import time
import glob
import sys
import pdb

# define HTML class
class HTML:
    def __init__(self, title, reflesh=0):
        self.title = title

        self.doc = dominate.document(title=title)
        if reflesh > 0:
            with self.doc.head:
                meta(http_equiv="reflesh", content=str(reflesh))


    def add_header(self, str):
        with self.doc:
            h3(str)

    def add_table(self, border=1):
        self.t = table(border=border, style="table-layout: fixed;")
        self.doc.add(self.t)

    def add_images(self, ims, txts, width=400):
        self.add_table()
        with self.t:
            with tr():
                for im, txt in zip(ims, txts):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(href=im):
                                img(style="width:%dpx" % width, src=im)
                            br()
                            p(txt)

    def save(self):
        html_file = '%s.html' % self.title
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()


def main():

#    name = 'disentangled_shuffle_resnet_9blocks_sigmoid_A100_TV1_lr0.0002'
    name = sys.argv[1]

    root = '/home/xyang/UTS/Data/Haze/D-HAZY/NYU/results/'+name+'/test_latest/images/'

    web_name = 'Dehaze_'+name
    suffix = ['_Hazy', '_dcp_radiance-refinedt', '_DehazeNet', '_Haze-free', '_real_B']
#    folder = ['/home/xyang/UTS/Data/Haze/D-HAZY/NYU/results/'+name+'/test_latest/images/',
#              'DCP/'+name+'/',   
#              'DehazeNet/'+name+'/',   
#              '/home/xyang/UTS/Data/Haze/D-HAZY/NYU/results/'+name+'/test_latest/images/',
#              '/home/xyang/UTS/Data/Haze/D-HAZY/NYU/results/'+name+'/test_latest/images/']
    folder = ['/static/'+name+'/images/', 
              '/static/DCP/',
              '/static/DehazeNet/',
              '/static/'+name+'/images/', 
              '/static/'+name+'/images/']

    assert len(suffix) == len(folder)

    img_names = glob.glob(root+'/*_Hazy_Hazy.png')
    max_num = 100
    win_size = 256

    # create website
    webpage = HTML(web_name)

    for i, img_name in enumerate(img_names):
        if i >= max_num:
            break

        img_name = os.path.basename(img_name)
        img_path = []
        for j in range(len(suffix)):
            img_path.append(os.path.join(folder[j], img_name.replace('_Hazy.png', suffix[j]+'.png')))

        webpage.add_header(img_name)
        webpage.add_images(img_path, suffix, win_size)

    webpage.save()

##################### Page 2####################

    web_name = 'Transmission_'+name
    suffix = ['_Hazy', '_dcp_refinedt', '_transmition', '_Estimate_depth', '_real_depth']
#    folder = ['/home/xyang/UTS/Data/Haze/D-HAZY/NYU/results/'+name+'/test_latest/images/',
#              '/home/xyang/Downloads/GAN/DCP/'+name+'/',   
#              '/home/xyang/Downloads/GAN/DehazeNet/'+name+'/',   
#              '/home/xyang/UTS/Data/Haze/D-HAZY/NYU/results/'+name+'/test_latest/images/',
#              '/home/xyang/UTS/Data/Haze/D-HAZY/NYU/results/'+name+'/test_latest/images/']
    folder = ['/static/'+name+'/images/', 
              '/static/DCP/',
              '/static/DehazeNet/',
              '/static/'+name+'/images/', 
              '/static/'+name+'/images/']

    assert len(suffix) == len(folder)

    img_names = glob.glob(root+'/*_Hazy_Hazy.png')
    max_num = 100
    win_size = 256

    # create website
    webpage = HTML(web_name)

    for i, img_name in enumerate(img_names):
        if i >= max_num:
            break

        img_name = os.path.basename(img_name)
        img_path = []
        for j in range(len(suffix)):
            img_path.append(os.path.join(folder[j], img_name.replace('_Hazy.png', suffix[j]+'.png')))

        webpage.add_header(img_name)
        webpage.add_images(img_path, suffix, win_size)

    webpage.save()


if __name__ == '__main__':
    main()
