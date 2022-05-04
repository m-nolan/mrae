import torch
import torch.nn as nn
import torch.nn.functional as F

class MRAE(nn.Module):

    def __init__(self):
        super().init()
        pass

    def forward(self,src):
        pass

    def get_checkpoint(self):
        pass

    def load_from_checkpoint(self,checkpoint):
        # create model from checkpoint file
        pass

class RAE_block(nn.Module):

    def __init__(self):
        super().init()
        # initalize models
        self.rand_samp = True
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.block_out = None

    def forward(self,src):
        # forward pass
        encoder_out = self.encoder(src)
        decoder_out = self.decoder(encoder_out)
        return self.block_out(decoder_out)

    def sample_generator_ic(self, mean, logvar):
        if self.rand_samp:
            return sample_gaussian(mean, logvar)
        else:
            return mean

class Encoder(nn.Module):

    def __init__(self):
        super().init()

    def forward(self,input):
        pass

class Decoder(nn.Module):

    def __init__(self):
        super().init()

    def forward(self,input):
        pass

def sample_gaussian(mean, logvar):
    # Generate noise from standard gaussian
    eps = torch.randn(mean.shape, requires_grad=False, dtype=torch.float32).to(torch.get_default_dtype()).to(mean.device)
    # Scale and shift by mean and standard deviation
    return torch.exp(logvar*0.5)*eps + mean
