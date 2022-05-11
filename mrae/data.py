from collections import namedtuple
import torch
import torch.nn as nn

# output collection structure (named tuple, good for potential DDP)
mrae_output = namedtuple(
    'mrae_output',
    [
        'output',
        'block_output',
        'hidden',
        'decoder_ic',
        'decoder_ic_kl_div',
        'decoder_l2'
        ]
)

# dataloader classes

