from collections import namedtuple
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import mrae
from .data import mrae_output

# loss collection structure
mrae_loss = namedtuple(
    'mrae_loss',
    [
        'output_loss',
        'block_loss',
        'kl_div',
        'l2'
    ]
)
class MRAEObjective(nn.Module):

    def __init__(self,kl_div_scale_max,l2_scale_max,max_at_epoch=100):
        super().__init__()
        # these values are updated by the scheduler during model optimization
        self.kl_div_scale_max = kl_div_scale_max
        self.l2_scale_max = l2_scale_max
        self.max_at_epoch = max_at_epoch

        # initialize scale at zero to begin training
        self.kl_div_scale = 0.
        self.l2_scale = 0.

    def forward(self, mrae_output:mrae_output, target:torch.Tensor, block_target:torch.Tensor):
        out = mrae_output.output
        block_out = mrae_output.block_output
        batch_size, sequence_length, num_ch = target.size()
        num_blocks = block_target.size(-1)
        
        output_mse = tensor_3d_mse(out, target)
        block_mse = tensor_3d_mse(block_out, block_target)

        output_objective = output_mse
        block_objective = block_mse + \
            self.kl_div_scale * mrae_output.decoder_ic_kl_div + \
            self.l2_scale * mrae_output.decoder_l2
        
        return output_objective, block_objective

    def step(self,epoch_idx):
        # update objective regularization term scalars according to epoch count
        # should this object be keeping count of the current epoch?
        epoch = epoch_idx + 1 # with zero-indexing, the first epoch wouldn't iterate this
        self.kl_div_scale = self.kl_div_scale_max * min(1, epoch/self.max_at_epoch)
        self.l2_scale = self.l2_scale_max * min(1, epoch/self.max_at_epoch)

# class MRAEScheduler(ReduceLROnPlateau):

#     def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
#                 verbose=False, threshold=1e-4, threshold_mode='rel',
#                 cooldown=0, min_lr=0, eps=1e-8):
#         super().__init__(optimizer=optimizer, mode=mode, factor=factor,
#             patience=patience, verbose=verbose, threshold=threshold,
#             threshold_mode=threshold_mode, cooldown=cooldown, min_lr=min_lr,
#             eps=eps)

#     def step(self, val_obj, epoch=None):
#         val_obj = float(val_obj)
#         if epoch is None:
#             epoch = self.last_epoch

# class LFADS_Scheduler(ReduceLROnPlateau):
    
#     def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
#                  verbose=False, threshold=1e-4, threshold_mode='rel',
#                  cooldown=0, min_lr=0, eps=1e-8):
        
#         super(LFADS_Scheduler, self).__init__(optimizer=optimizer, mode=mode, factor=factor, patience=patience,
#                                               verbose=verbose, threshold=threshold, threshold_mode=threshold_mode,
#                                               cooldown=cooldown, min_lr=min_lr, eps=eps)
        
        
#     def step(self, metrics, epoch=None):
#         # convert `metrics` to float, in case it's a zero-dim Tensor
#         current = float(metrics)
#         if epoch is None:
#             epoch = self.last_epoch = self.last_epoch + 1
#         self.last_epoch = epoch

#         if self.is_better(current, self.best):
#             self.best = current
#             self.num_bad_epochs = 0
#         else:
#             self.num_bad_epochs += 1

#         if self.in_cooldown:
#             self.cooldown_counter -= 1
#             self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

#         if self.num_bad_epochs > self.patience:
#             self._reduce_lr(epoch)
#             self.cooldown_counter = self.cooldown
#             self.num_bad_epochs = 0
#             self.best = self.mode_worse

def tensor_3d_mse(output, target):
    """tensor_3d_mse

    return mean squared error between model ouput and target tensors.

    mean computed over all 3 initial tensor dimensions. output and target must be equal in array size.

    Args:
        output (torch.Tensor): model output tensor.
        target (torch.Tensor): output target tensor

    Returns:
        mse (torch.Tensor): mean squared error between input tensors. Tensor size equal to input dimensions beyond first three tensor dimensions.
    """
    assert output.size() == target.size()
    return (output - target).pow(2).mean(dim=(0,1,2))

def backward_on_block_params(obj,block,retain_graph=False):
    """backward_on_block_params

    run backwards diff pass from objective obj on parameters from a give model block block.

    Args:
        obj (torch.Tensor): pytorch model objective output
        block (nn.Module): pytorch model block
        retain_graph (bool, optional): retain_graph option in obj.backward(). Defaults to False.
    """
    obj.backward(
        inputs=[p for p in block.parameters() if p.requires_grad],
        retain_graph=retain_graph
    )