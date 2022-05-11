import torch
import torch.nn as nn

import mrae
from .data import mrae_output

class MRAEObjective(nn.Module):

    def __init__(self,kl_div_scale,l2_scale):
        super().__init__()
        # these values are updated by the scheduler during model optimization
        self.kl_div_scale = kl_div_scale
        self.l2_scale = l2_scale

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