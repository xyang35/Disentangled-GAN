import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

"""
Stage 1
    Given ground truth depth, try to get good haze-free image from hazy image
    and use the output and gt depth to recover hazy image
"""


class ReconContModel(BaseModel):
    def name(self):
        return 'ReconContModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # define tensors
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_C = self.Tensor(opt.batchSize, opt.depth_nc,
                                   opt.fineSize, opt.fineSize)

        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_B = networks.define_D(opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)
            self.netD_A = networks.define_D(opt.input_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)

            # pretrained VGG
            self.vgg = networks.define_VGG(gpu_ids=self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionContent = torch.nn.MSELoss()

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD_B)
            networks.print_network(self.netD_A)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        # depth image
        input_C = input['C']
        input_C = (input_C + 1) / 2.0    # rescale to [0, 1]
        self.input_C.resize_(input_C.size()).copy_(input_C)

        if self.opt.depth_reverse:
            self.input_C = 1 - self.input_C

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.fake_B = self.netG.forward(self.real_A)
        self.real_B = Variable(self.input_B)

        # reconstruct A based on optical model
        self.real_C = Variable(self.input_C, requires_grad=False)
        self.fake_A = util.synthesize_matting(self.fake_B, self.real_C)

    # no backprop gradients
    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG.forward(self.real_A)
        self.real_B = Variable(self.input_B, volatile=True)

        # reconstruct A based on optical model
        self.real_C = Variable(self.input_C, volatile=True)
        self.fake_A = util.synthesize_matting(self.fake_B, self.real_C)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake):
        # Fake
        # stop backprop to the generator by detaching fake_B
        pred_fake = netD.forward(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        pred_real = netD.forward(real)
        loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        loss_D = (loss_D_fake + loss_D_real) * 0.5

        loss_D.backward()
        return loss_D
    
    def backward_D_B(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_B, self.fake_B)

    def backward_D_A(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_A, self.fake_A)

    def backward_G(self):
        # First, G(A) should fake the discriminator
        pred_fake_B = self.netD_B.forward(self.fake_B)
        self.loss_G_B = self.criterionGAN(pred_fake_B, True)

        # Second, reconstruction should fake the discriminator
        pred_fake_A = self.netD_A.forward(self.fake_A)
        self.loss_G_A = self.criterionGAN(pred_fake_A, True)

        # Third, L1 loss for reconstruction
        self.loss_G_L1 = self.criterionL1(self.fake_A, self.real_A) * self.opt.lambda_A

        # Forth, content loss
        feat_real = self.vgg.forward(self.real_A)
        feat_fake = self.vgg.forward(self.fake_B)
        feat_real = Variable(feat_real.data, requires_grad=False)
        self.loss_Content = self.criterionContent(feat_fake, feat_real) * self.opt.lambda_Content

        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_G_L1 + self.loss_Content

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()

        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_B', self.loss_G_B.data[0]),
                            ('G_A', self.loss_G_A.data[0]),
                            ('G_L1', self.loss_G_L1.data[0]),
                            ('G_Content', self.loss_Content.data[0]),
                            ('D_B', self.loss_D_B.data[0]),
                            ('D_A', self.loss_D_A.data[0])
                            ])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        input_C = (self.input_C - 0.5) * 2.0    # rescale back to [-1, 1] before tensor2im
        depth = util.tensor2im(input_C)
        fake_A = util.tensor2im(self.fake_A.data)
        return OrderedDict([('Hazy', real_A), ('Haze-free', fake_B), ('depth', depth), ('recover', fake_A), ('real_B', real_B)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D_A.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_D_B.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
