import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import random

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def get_transform(opt):
    transform_list = []
    if opt.resize_or_crop == 'resize_and_crop':
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))
        if opt.isTrain:
            transform_list.append(transforms.RandomCrop(opt.fineSize))
        else:    # no random crop in testing time
            transform_list.append(transforms.CenterCrop(opt.fineSize))

    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize)))
        if opt.isTrain:
            transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'disentangled':
        # my version of tranform
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize)))
        if opt.isTrain:
            transform_list.append(transforms.Lambda(
                lambda img: __crop_width_random(img, opt.fineSize)))
        else:
            # center cropping at test time
            transform_list.append(transforms.Lambda(
                lambda img: __crop_width_center(img, opt.fineSize)))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)

def __crop_width_center(img, target_width):
    ow, oh =  img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow) / 4 * 4   # to ensure factor of 4 for conv-deconv

    x1 = int(round((ow - w) / 2.))
    y1 = int(round((oh - h) / 2.))
    return img.crop((x1, y1, x1 + w, y1 + h))

def __crop_width_random(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow) / 4 * 4   # to ensure factor of 4 for conv-deconv

    x1 = random.randint(0, ow-w)
    y1 = random.randint(0, oh-h)
    return img.crop((x1, y1, x1 + w, y1 + h))
