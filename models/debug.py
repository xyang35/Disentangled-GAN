import torch
from torch.autograd import Variable
from collections import OrderedDict
import util.util as util
from .base_model import BaseModel
from . import networks


class DebugModel(BaseModel):
    def name(self):
        return 'DebugModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)

        self.netG = networks.define_G(input_nc=opt.input_nc, output_nc=opt.output_nc,
                                      ngf=opt.ngf, which_model_netG=opt.which_model_depth,
                                      gpu_ids=self.gpu_ids, non_linearity=opt.non_linearity,
                                      pooling=opt.pooling)
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)

        if self.isTrain:
            # pretrained VGG
            self.vgg = networks.define_VGG(gpu_ids=self.gpu_ids)

            self.grad_clip = opt.grad_clip
            self.old_lr = opt.lr
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionMSE = torch.nn.MSELoss()
            self.criterionTV = networks.TVLoss()
            self.optimizer = torch.optim.Adam(self.netG.parameters(),
                                              lr=opt.lr, betas=(opt.beta1, 0.999))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        print('-----------------------------------------------')

    def set_input(self, input):
        input_A = input['A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.image_paths = input['A_paths']
        input_B = input['B']
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['B_paths']

    def forward_backward(self):
        self.real_A = Variable(self.input_A)
        self.fake_B = self.netG.forward(self.real_A)
        self.real_B = Variable(self.input_B)

        # important to rescale ground truth from [-1,1] to [0,1]
        B_rescaled = (self.real_B + 1) / 2.
        self.loss_MSE = self.criterionMSE(self.fake_B.expand_as(self.real_B), B_rescaled) * self.opt.lambda_A

        # perceptual loss
        self.loss_perceptual = 0
        if not self.opt.lambda_perceptual == 0:
            feat_real = self.vgg.forward(B_rescaled)
            feat_fake = self.vgg.forward(self.fake_B.expand_as(self.real_B))
            for i in range(len(feat_fake)):
                f = Variable(feat_real[i].data, requires_grad=False)
                self.loss_perceptual += self.criterionL1(feat_fake[i], f)
    
            self.loss_perceptual *= self.opt.lambda_perceptual

        # Total variance loss
        self.loss_TV = self.criterionTV(self.fake_B) * self.opt.lambda_TV

        self.loss = self.loss_MSE + self.loss_perceptual + self.loss_TV
        self.loss.backward()

        # gradient clip helps prevent the exploding gradient problem
        if not self.grad_clip == -1:
            torch.nn.utils.clip_grad_norm(self.netG.parameters(), self.grad_clip)

    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG.forward(self.real_A)
        self.real_B = Variable(self.input_B, volatile=True)

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.forward_backward()
        self.optimizer.step()

    def get_current_errors(self):
        if self.opt.lambda_perceptual:
            return OrderedDict([('L_MSE', self.loss_MSE.data[0]), ('L_perceptual', self.loss_perceptual.data[0]), ('L_TV', self.loss_TV.data[0])])
        else:
            return OrderedDict([('L_MSE', self.loss_MSE.data[0]), ('L_TV', self.loss_TV.data[0])])

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
