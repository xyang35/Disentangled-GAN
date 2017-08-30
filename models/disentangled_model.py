import numpy as np
import torch
import os
import itertools
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class DisentangledModel(BaseModel):
    def name(self):
        return 'DisentangledModel'

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
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids,
                                      non_linearity=opt.non_linearity, pooling=opt.pooling)
        self.netDepth = networks.define_G(input_nc=opt.input_nc, output_nc=1, ngf=6,
                                      which_model_netG=opt.which_model_depth, 
                                      gpu_ids=self.gpu_ids, non_linearity=opt.non_linearity, pooling=opt.pooling)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            self.load_network(self.netDepth, 'Depth', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            self.fake_B_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionTV = networks.TVLoss()

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters(), self.netDepth.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        networks.print_network(self.netDepth)
        if self.isTrain:
            networks.print_network(self.netD)
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
        self.input_C.resize_(input_C.size()).copy_(input_C)

        if self.opt.depth_reverse:
            self.input_C = - self.input_C

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.fake_B = self.netG.forward(self.real_A)
        self.real_B = Variable(self.input_B)
        self.depth = self.netDepth.forward(self.real_A)

        # clip with 0.9
        self.depth = torch.clamp(self.depth, max=0.9)

        # recover B according to depth
        self.fake_B2 = util.reverse_matting(self.real_A, self.depth)

        # reconstruct A based on optical model
        self.fake_A = util.synthesize_matting(self.fake_B, self.depth)

    # no backprop gradients
    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG.forward(self.real_A)
        self.real_B = Variable(self.input_B, volatile=True)
        self.depth = self.netDepth.forward(self.real_A)

        # clip with 0.9
        self.depth = torch.clamp(self.depth, max=0.9)

        # recover B according to depth
        self.fake_B2 = util.reverse_matting(self.real_A, self.depth)

        # reconstruct A based on optical model
        self.fake_A = util.synthesize_matting(self.fake_B, self.depth)

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
        loss_D = loss_D_fake + loss_D_real

        loss_D.backward()
        return loss_D
    
    def backward_D(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D = self.backward_D_basic(self.netD, self.real_B, self.fake_B)

    def backward_G(self):
        # First, G(A) should fake the discriminator
        pred_fake_B = self.netD.forward(self.fake_B)
        self.loss_G_B = self.criterionGAN(pred_fake_B, True)

        # Second, L1 loss for reconstruction
        self.loss_G_L1 = self.criterionL1(self.fake_A, self.real_A) * self.opt.lambda_A

        # Third, total variance loss
        self.loss_TV = self.criterionTV(self.depth) * self.opt.lambda_TV

        self.loss_G = self.loss_G_L1 + self.loss_G_B + self.loss_TV

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_B', self.loss_G_B.data[0]),
                            ('G_L1', self.loss_G_L1.data[0]),
                            ('D', self.loss_D.data[0]),
                            ('TV', self.loss_TV.data[0])
                            ])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        fake_depth = util.tensor2im(self.depth.data)
        real_B = util.tensor2im(self.real_B.data)
        real_depth = util.tensor2im(self.input_C)
        fake_A = util.tensor2im(self.fake_A.data)
        fake_B2 = util.tensor2im(self.fake_B2.data)
        return OrderedDict([('Hazy', real_A), ('Haze-free', fake_B), ('Haze-free-depth', fake_B2), ('Estimate_depth', fake_depth), 
            ('recover', fake_A), ('real_depth', real_depth), ('real_B', real_B)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netDepth, 'Depth', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
