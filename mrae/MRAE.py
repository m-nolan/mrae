import torch
import torch.nn as nn
import torch.nn.functional as F
from . import rnn

from warnings import warn

class MRAE(nn.Module):

    def __init__(self):
        super().__init__()
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
        super().__init__()
        # initalize models
        self.rand_samp  = True
        self.encoder    = Encoder()
        self.decoder    = Decoder()
        self.block_out  = None

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

    def __init__(self, input_size, output_size, hidden_size, dropout, num_layers=1, bidirectional=True):
        super().__init__()
        self.input_size     = input_size
        self.output_size    = output_size
        self.hidden_size    = hidden_size
        self.dropout        = dropout
        self.num_layers     = num_layers
        self.bidirectional  = bidirectional
        
        self.rnn    = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )

        self.dropout_layer = nn.Dropout(dropout)

        bidir_scale = 2 if bidirectional else 1
        gru_output_size = bidir_scale * num_layers * hidden_size
        if gru_output_size == 2 * output_size:
            self.linear_out = nn.Identity()
        else:
            self.linear_out = nn.Linear(in_features=gru_output_size,out_features=2*output_size)

    def forward(self,input):
        batch_size, seq_len, input_size = input.size()
        assert input_size == self.input_size
        rnn_out, rnn_n = self.rnn(input)
        #TODO: add rnn output clip
        rnn_n = self.dropout_layer(rnn_n)
        rnn_n = rnn_n.permute(1,0,2).reshape(batch_size,-1) # [batch_size, num_layers * bidir_scale * hidden_size]
        param_out = self.linear_out(rnn_n)
        mean, logvar = torch.split(param_out,self.output_size,dim=-1)
        return mean, logvar

class Decoder(nn.Module):

    def __init__(self, hidden_size, dropout, num_layers=1, bidirectional=False):
        super().__init__()
        self.input_size     = 1 # only >1 if there's a controller, not used in MRAE as-is
        self.hidden_size    = hidden_size # equal to size of encoder outputs
        self.dropout        = dropout
        if num_layers > 1:
            warn('Multi-layer decoder not currently supported. Defaulting to single layer.')
        self.num_layers     = 1
        if bidirectional:
            warn('Bidirectional decoder not currently supported. Defaulting to unidirectional.')
        self.bidirectional  = False

        self.rnn    = rnn.GRU_Modified(self.input_size,self.hidden_size)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self,input,h0):
        gen_out = self.rnn(input,h0=h0)
        gen_out = self.dropout_layer(gen_out)
        return gen_out

    def gen_input(self,batch_size,seq_len):
        return torch.zeros(batch_size,seq_len,self.input_size)

def sample_gaussian(mean, logvar):
    """
    sample_gaussian(mean, logvar)

    Generates a tensor of samples from an input parameterization.
    Used for generating random latent samples when using the VAE reparameterization trick.

    Args:
        mean (torch.Tensor): tensor of means
        logvar (torch.Tensor): tensor of log variances

    Returns:
        sample (torch.Tensor): tensor of gaussian samples
    """
    # Generate noise from standard gaussian
    eps = torch.randn(mean.shape, requires_grad=False, dtype=torch.float32).to(torch.get_default_dtype()).to(mean.device)
    # Scale and shift by mean and standard deviation
    return torch.exp(logvar*0.5)*eps + mean
