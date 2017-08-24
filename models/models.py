
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'cycle_gan':
        assert(opt.dataset_mode == 'unaligned')
        from .cycle_gan_model import CycleGANModel
        model = CycleGANModel()
    elif opt.model == 'pix2pix':
        assert(opt.dataset_mode == 'aligned')
        from .pix2pix_model import Pix2PixModel
        model = Pix2PixModel()
    elif opt.model == 'recon':
        assert(opt.dataset_mode == 'depth')
        from .recon_model import ReconModel
        model = ReconModel()
    elif opt.model == 'recon_cont':
        assert(opt.dataset_mode == 'depth')
        from .recon_content_model import ReconContModel
        model = ReconContModel()
    elif opt.model == 'disentangled':
        assert(opt.dataset_mode == 'depth')
        from .disentangled_model import DisentangledModel
        model = DisentangledModel()
    elif opt.model == 'disentangled2':
        assert(opt.dataset_mode == 'depth')
        from .disentangled_model2 import DisentangledModel
        model = DisentangledModel()
    elif opt.model == 'test':
        assert(opt.dataset_mode == 'single')
        from .test_model import TestModel
        model = TestModel()
    elif opt.model == 'debug':
        from .debug import DebugModel
        model = DebugModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
