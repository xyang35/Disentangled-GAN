import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
#from vgg16 import Vgg16
import pdb
###############################################################################
# Functions
###############################################################################


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


# def define_VGG(pretrained=True, gpu_ids=[]):
#     vgg = Vgg16()

#     if pretrained:
#         vgg.load_state_dict(torch.load('models/vgg16.weight'))

#     if len(gpu_ids) > 0:
#         vgg.cuda(device_id=gpu_ids[0])
#     return vgg


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, gpu_ids=[], non_linearity=None, pooling=False, n_layers=3, filtering=None):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids, non_linearity=non_linearity, filtering=filtering)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids, non_linearity=non_linearity, filtering=filtering)
    elif which_model_netG == 'aod':
        netG = AODNetGenerator(input_nc, output_nc, ngf, gpu_ids=gpu_ids, non_linearity=non_linearity, pooling=pooling, filtering=filtering, norm_layer=norm_layer)
    elif which_model_netG == 'air':
        netG = AirGenerator(gpu_ids=gpu_ids, n_layers=n_layers)
    elif which_model_netG == 'resnet6_depth':
        netG = ResnetDepthGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids, non_linearity=non_linearity, filtering=filtering)
    elif which_model_netG == 'resnet9_depth':
        netG = ResnetDepthGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids, non_linearity=non_linearity, filtering=filtering)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.to(gpu_ids[0])
    netG.apply(weights_init)
    return netG


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'multi':
        netD = MultiDiscriminator(input_nc, ndf, n_layers=n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    if use_gpu:
        netD.to(gpu_ids[0])
    netD.apply(weights_init)
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################

class TVLoss(nn.Module):
    '''
    Define Total Variance Loss for images
    which is used for smoothness regularization
    '''

    def __init__(self):
        super(TVLoss, self).__init__()

    def __call__(self, input):
        # Tensor with shape (n_Batch, C, H, W)
        origin = input[:, :, :-1, :-1]
        right = input[:, :, :-1, 1:]
        down = input[:, :, 1:, :-1]

        tv = torch.mean(torch.abs(origin-right)) + torch.mean(torch.abs(origin-down))
        return tv * 0.5


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


# Define an airlight Net
class AirGenerator(nn.Module):
    def __init__(self, gpu_ids=[], n_layers=2):
        super(AirGenerator, self).__init__()
        self.gpu_ids = gpu_ids
        self.n_layers = n_layers

        model = []
        for i in range(n_layers-1):
            model += [nn.ReflectionPad2d(1),
                      nn.Conv2d(3, 3, kernel_size=3, padding=0),
                      nn.BatchNorm2d(3),
                      nn.ReLU(True)]

            model += [nn.MaxPool2d(kernel_size=2, padding=0)]

        # global pooling at the last layer
        model += [nn.ReflectionPad2d(1),
                  nn.Conv2d(3, 3, kernel_size=3, padding=0),
                  nn.Sigmoid(),
                  nn.AdaptiveMaxPool2d(1)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Define AOD-Net
class AODNetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=1, ngf=6, norm_layer=nn.BatchNorm2d, gpu_ids=[], non_linearity=None, pooling=False, filtering=None, r=10, eps=1e-3):
        super(AODNetGenerator, self).__init__()
        self.input_nc = input_nc
        self.gpu_ids = gpu_ids
        self.pooling = pooling
        self.filtering = filtering
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        self.non_linearity = non_linearity
        if non_linearity is None:
            last_act = nn.Tanh()
        elif non_linearity == 'BReLU':
            last_act = BReLU(0.95,0.05,0.95,0.05,True)
        elif non_linearity == 'ReLU':
            last_act = nn.ReLU(True)
        elif non_linearity == 'sigmoid':
            last_act = nn.Sigmoid()
        elif non_linearity == 'linear':
            last_act = None
        else:
            print(non_linearity)
            raise NotImplementedError

        model = [nn.Conv2d(input_nc, ngf, kernel_size=1, padding=0, bias=use_bias),
#                 nn.BatchNorm2d(ngf),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        if pooling:
            model += [nn.MaxPool2d(kernel_size=2, padding=0),
                      nn.Upsample(scale_factor=2, mode='nearest')]

        model += [ConcatBlock(ngf, pooling, norm_layer)]

        model += [nn.ReflectionPad2d(1),
                  nn.Conv2d(4*ngf, output_nc, kernel_size=3, padding=0)]

        if last_act is not None:
            model += [last_act]
        self.model = nn.Sequential(*model)    # nn.Sequential only accepts a single input.

        if filtering is not None:
            if filtering == 'max':
                self.last_layer = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)
            elif filtering == 'guided':
                self.last_layer = GuidedFilter(r=r, eps=eps)

    def forward(self, input):
        if self.filtering == 'guided':
            # rgb2gray
            guidance = 0.2989 * input[:,0,:,:] + 0.5870 * input[:,1,:,:] + 0.1140 * input[:,2,:,:]
            # rescale to [0,1]
            guidance = (guidance + 1) / 2
            guidance = torch.unsqueeze(guidance, dim=1)

        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            pre_filter = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
            if self.non_linearity is None:
                # rescale to [0,1]
                pre_filter = (pre_filter + 1) / 2

            if self.filtering is not None:
                if self.filtering == 'guided':
                    return pre_filter, self.last_layer(guidance, pre_filter)
                else:
                    return pre_filter, nn.parallel.data_parallel(self.last_layer, pre_filter, self.gpu_ids)
            else:
                return None, pre_filter
        else:
            pre_filter = self.model(input)
            if self.non_linearity is None:
                # rescale to [0,1]
                pre_filter = (pre_filter + 1) / 2

            if self.filtering is not None:
                if self.filtering == 'guided':
                    return pre_filter, self.last_layer(guidance, pre_filter)
                else:
                    return pre_filter, self.last_layer(pre_filter)
            else:
                return None, pre_filter



# Guided image filtering for grayscale images
class GuidedFilter(nn.Module):
    def __init__(self, r=40, eps=1e-3, tensor=torch.cuda.FloatTensor):    # only work for gpu case at this moment
        super(GuidedFilter, self).__init__()
        self.r = r
        self.eps = eps
        self.tensor = tensor

        self.boxfilter = nn.AvgPool2d(kernel_size=2*self.r+1, stride=1,padding=self.r)

    def forward(self, I, p):
        """
        I -- guidance image, should be [0, 1]
        p -- filtering input image, should be [0, 1]
        """
        
        N = self.boxfilter(Variable(self.tensor(p.size()).fill_(1),requires_grad=False))

        mean_I = self.boxfilter(I) / N
        mean_p = self.boxfilter(p) / N
        mean_Ip = self.boxfilter(I*p) / N
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = self.boxfilter(I*I) / N
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + self.eps)
        b = mean_p - a * mean_I
        mean_a = self.boxfilter(a) / N
        mean_b = self.boxfilter(b) / N

        return mean_a * I + mean_b


class ConcatBlock(nn.Module):
    def __init__(self, input_nc, pooling=False, norm_layer=nn.BatchNorm2d):
        super(ConcatBlock, self).__init__()
        self.conv_block1 = self.build_block(input_nc, 3, input_nc, pooling, norm_layer)
        self.conv_block2 = self.build_block(2*input_nc, 5, input_nc, pooling, norm_layer)
        self.conv_block3 = self.build_block(2*input_nc, 7, input_nc, pooling, norm_layer)

    def build_block(self, input_nc, kernel_size, output_nc, pooling, norm_layer):
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(kernel_size/2),
                 nn.Conv2d(input_nc, output_nc, kernel_size=kernel_size, padding=0, bias=use_bias),
                 norm_layer(output_nc),
                 nn.ReLU(True)]

        if pooling:
            model += [nn.MaxPool2d(kernel_size=2, padding=0),
                      nn.Upsample(scale_factor=2, mode='nearest')]

        return nn.Sequential(*model)

    def forward(self, conv1):
        # naming is according to the paper
        conv2 = self.conv_block1(conv1)
        concat1 = torch.cat([conv1, conv2], 1)

        conv3 = self.conv_block2(concat1)
        concat2 = torch.cat([conv2, conv3], 1)

        conv4 = self.conv_block3(concat2)
        concat3 = torch.cat([conv1,conv2,conv3,conv4],1)

        return concat3

class BReLU(nn.Module):
    def __init__(self, up_thred, down_thred, up_value, down_value, inplace=False):
        super(BReLU, self).__init__()
        self.up_threshold = up_thred
        self.down_threshold = down_thred
        self.up_value = up_value
        self.down_value = down_value
        self.inplace = inplace

    def forward(self, input):
        temp = nn.functional.threshold(input, self.down_threshold, self.down_value, self.inplace)
        return -nn.functional.threshold(-temp, -self.up_threshold, -self.up_value, self.inplace)

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' \
                + str(self.up_threshold) \
                + str(self.down_threshold) \
                + ', ' + str(self.up_value) \
                + ', ' + str(self.down_value) \
                + inplace_str + ')'

# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d


        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
#            model += [nn.Upsample(scale_factor=2, mode='nearest'),
#                      nn.Conv2d(ngf * mult, int(ngf * mult / 2), 3, padding=1, bias=use_bias),
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1, bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


class ResnetDepthGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect', non_linearity=None, filtering=None, r=10, eps=1e-3):
        assert(n_blocks >= 0)
        super(ResnetDepthGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.filtering=filtering
        self.non_linearity = non_linearity
        if non_linearity is None:
            last_act = nn.Tanh()
        elif non_linearity == 'BReLU':
            last_act = BReLU(0.95,0.05,0.95,0.05,True)
        elif non_linearity == 'ReLU':
            last_act = nn.ReLU(True)
        elif non_linearity == 'sigmoid':
            last_act = nn.Sigmoid()
        elif non_linearity == 'linear':
            last_act = None
        else:
            print(non_linearity)
            raise NotImplementedError

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
#            model += [nn.Upsample(scale_factor=2, mode='nearest'),
#                      nn.Conv2d(ngf * mult, int(ngf * mult / 2), 3, padding=1, bias=use_bias),
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1, bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if last_act is not None:
            model += [last_act]

        self.model = nn.Sequential(*model)

        if filtering is not None:
            if filtering == 'max':
                self.last_layer = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)
            elif filtering == 'guided':
                self.last_layer = GuidedFilter(r=r, eps=eps)

    def forward(self, input):
        if self.filtering == 'guided':
            # rgb2gray
            guidance = 0.2989 * input[:,0,:,:] + 0.5870 * input[:,1,:,:] + 0.1140 * input[:,2,:,:]
            # rescale to [0,1]
            guidance = (guidance + 1) / 2
            guidance = torch.unsqueeze(guidance, dim=1)

        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            pre_filter = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
            if self.non_linearity is None:
                # rescale to [0,1]
                pre_filter = (pre_filter + 1) / 2

            if self.filtering is not None:
                if self.filtering == 'guided':
                    return pre_filter, self.last_layer(guidance, pre_filter)
                else:
                    return pre_filter, nn.parallel.data_parallel(self.last_layer, pre_filter, self.gpu_ids)
            else:
                return None, pre_filter
        else:
            pre_filter = self.model(input)
            if self.non_linearity is None:
                # rescale to [0,1]
                pre_filter = (pre_filter + 1) / 2

            if self.filtering is not None:
                if self.filtering == 'guided':
                    return pre_filter, self.last_layer(guidance, pre_filter)
                else:
                    return pre_filter, self.last_layer(pre_filter)
            else:
                return None, pre_filter


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias=use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[], non_linearity=None, filtering=None, r=10, eps=1e-3):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids
        self.filtering=filtering
        self.non_linearity = non_linearity
        if non_linearity is None:
            last_act = nn.Tanh()
        elif non_linearity == 'BReLU':
            last_act = BReLU(0.95,0.05,0.95,0.05,True)
        elif non_linearity == 'ReLU':
            last_act = nn.ReLU(True)
        elif non_linearity == 'sigmoid':
            last_act = nn.Sigmoid()
        elif non_linearity == 'linear':
            last_act = None
        else:
            print(non_linearity)
            raise NotImplementedError

        # currently support only input_nc == output_nc
#        assert(input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer, non_linearity=last_act)

        self.model = unet_block

        if filtering is not None:
            if filtering == 'max':
                self.last_layer = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)
            elif filtering == 'guided':
                self.last_layer = GuidedFilter(r=r, eps=eps)

    def forward(self, input):
        if self.filtering == 'guided':
            # rgb2gray
            guidance = 0.2989 * input[:,0,:,:] + 0.5870 * input[:,1,:,:] + 0.1140 * input[:,2,:,:]
            # rescale to [0,1]
            guidance = (guidance + 1) / 2
            guidance = torch.unsqueeze(guidance, dim=1)

        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            pre_filter = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
            if self.non_linearity is None:
                # rescale to [0,1]
                pre_filter = (pre_filter + 1) / 2

            if self.filtering is not None:
                if self.filtering == 'guided':
                    return pre_filter, self.last_layer(guidance, pre_filter)
                else:
                    return pre_filter, nn.parallel.data_parallel(self.last_layer, pre_filter, self.gpu_ids)
            else:
                return None, pre_filter
        else:
            pre_filter = self.model(input)
            if self.non_linearity is None:
                # rescale to [0,1]
                pre_filter = (pre_filter + 1) / 2

            if self.filtering is not None:
                if self.filtering == 'guided':
                    return pre_filter, self.last_layer(guidance, pre_filter)
                else:
                    return pre_filter, self.last_layer(pre_filter)
            else:
                return None, pre_filter


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, non_linearity=None):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # modify it temporally for depth output
        downconv = nn.Conv2d(3 if outermost else outer_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downconv]
#            up = [uprelu, upconv, nn.Tanh()]
            # modify it temporally for depth output
            up = [uprelu, upconv, non_linearity]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            y = self.model(x)
            print(y.size(), x.size())
            return torch.cat([y, x], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

# Defines the Multiscale-PatchGAN discriminator with the specified arguments.
class MultiDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=5, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(MultiDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # cannot deal with use_sigmoid=True case at thie moment
        assert(use_sigmoid == False)

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        scale1 = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, 3):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            scale1 += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        
        self.scale1 = nn.Sequential(*scale1)
        scale1_output = []
        scale1_output += [
            nn.Conv2d(ndf * nf_mult, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        scale1_output += [nn.Conv2d(ndf*nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]    # compress to 1 channel
        self.scale1_output = nn.Sequential(*scale1_output)

        scale2 = []
        nf_mult = nf_mult
        for n in range(3, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            scale2 += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        scale2 += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        scale2 += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            scale2 += [nn.Sigmoid()]

        self.scale2 = nn.Sequential(*scale2)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            scale1 = nn.parallel.data_parallel(self.scale1, input, self.gpu_ids)
            output1 = nn.parallel.data_parallel(self.scale1_output, scale1, self.gpu_ids)
            output2 = nn.parallel.data_parallel(self.scale2, scale1, self.gpu_ids)
        else:
            scale1 = self.scale1(input)
            output1 = self.scale1_output(scale1)
            output2 = self.scale2(scale1)
        
        return output1, output2
