import os
from torch.utils.tensorboard import SummaryWriter

def configure_tensorboard(proj_dir=None):
    tb_dir = os.path.join(proj_dir,'tensorboard')
    if os.path.exists(tb_dir):
        pass
    else:
        os.mkdir(tb_dir)
    writer = SummaryWriter(tb_dir)
    # TODO: add plotter function
    return writer
