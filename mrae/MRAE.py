import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from . import rnn
from .data import mrae_output
from .objective import backward_on_block_params

from warnings import warn

class MRAE(nn.Module):

    def __init__(self, input_size, encoder_size, decoder_size, num_blocks, 
                dropout, num_layers_encoder=1, bidirectional_encoder=True, 
                max_grad_norm=5.0, rand_samp=True, device='cpu'):
        super().__init__()
        self.input_size = input_size
        self.num_blocks = num_blocks
        self.device=device
        self.max_grad_norm = max_grad_norm
        
        # RAE blocks
        self.rae_blocks = nn.ModuleList(
            self.num_blocks * [
                RAE_block(
                    input_size=input_size,
                    encoder_size=encoder_size,
                    decoder_size=decoder_size,
                    dropout=dropout,
                    num_layers_encoder=num_layers_encoder,
                    bidirectional_encoder=bidirectional_encoder,
                    rand_samp=rand_samp
                )
            ]
        )
        # block output mixing layer. If this is a 1-block model, just use the block output.
        # otherwise, make a mixing layer that combines all block hidden state activity.
        if self.num_blocks > 1:
            self.block_hidden_mix = nn.Linear(
                in_features=num_blocks*decoder_size,
                out_features=input_size
            )
        else:
            self.block_hidden_mix = nn.Identity()

        self.to(self.device)

    def forward(self,input):
        batch_size, seq_len, input_size, n_band = input.shape
        assert input_size == self.input_size
        assert n_band == self.num_blocks

        block_output = []
        block_hidden = []
        block_dec_ic = []
        block_dec_ic_kl_div = []
        block_dec_l2 = []
        for b_idx in range(self.num_blocks):
            _b_o, _b_h, _b_d_ic, _b_kld = self.rae_blocks[b_idx](input[...,b_idx])
            _b_l2 = self.rae_blocks[b_idx].compute_decoder_l2()
            block_output.append(_b_o)
            block_hidden.append(_b_h)
            block_dec_ic.append(_b_d_ic)
            block_dec_ic_kl_div.append(_b_kld)
            block_dec_l2.append(_b_l2)
        block_output = torch.stack(block_output,dim=-1)
        block_hidden = torch.stack(block_hidden,dim=-1)
        block_dec_ic = torch.stack(block_dec_ic,dim=-1)
        block_dec_ic_kl_div = torch.stack(block_dec_ic_kl_div,dim=-1)
        block_dec_l2 = torch.stack(block_dec_l2,dim=-1)
        if self.num_blocks > 1:
            output = self.block_hidden_mix(block_hidden.reshape(batch_size,seq_len,-1))
        else:
            output = self.block_hidden_mix(block_output[...,0])

        return mrae_output(
            output=output, 
            block_output=block_output, 
            hidden=block_hidden, 
            decoder_ic=block_dec_ic,
            decoder_ic_kl_div=block_dec_ic_kl_div, 
            decoder_l2=block_dec_l2)

    def backward(self, output_obj, block_obj):
        for b_opt in self.block_opt:
            b_opt.zero_grad()
        if self.num_blocks > 1:
            self.output_opt.zero_grad()
        for b_idx in range(self.num_blocks):
            backward_on_block_params(
                block_obj[b_idx],
                self.rae_blocks[b_idx],
                retain_graph=True
            )
            torch.nn.utils.clip_grad_norm_(
                self.rae_blocks[b_idx].parameters(),
                max_norm=self.max_grad_norm
            )
        if self.num_blocks > 1:
            backward_on_block_params(
                output_obj,
                self.block_hidden_mix
            )
            torch.nn.utils.clip_grad_norm_(
                self.block_hidden_mix.parameters(),
                max_norm=self.max_grad_norm
            )

    def train(self, trail_dl, valid_dl, max_epochs=250, ):
        pass

    def eval(self, test_dl):
        pass

    def step_schedulers(self, output_obj, block_obj):
        # call this after validation step
        for b_idx in range(self.num_blocks):
            self.block_sch[b_idx].step(block_obj[b_idx])
        if self.num_blocks > 1:
            self.output_sch.step(output_obj)

    def get_optimizers(self):
        # default options for now
        # block optimizers
        block_opt = [
            torch.optim.Adam(
                [p for p in rae_block.parameters() if p.requires_grad]
            ) for rae_block in self.rae_blocks
        ]
        # output mixing layer optimizer - check if broken on 1-block instance
        output_opt = torch.optim.Adam(
            [p for p in self.block_hidden_mix.parameters() if p.requires_grad]
        ) if self.num_blocks > 1 else None # is there an empty optimizer base class to use instead?

        return output_opt, block_opt

    def get_schedulers(self):
        assert hasattr(self,'output_opt'), 'Output layer optimizer not found.'
        assert hasattr(self,'block_opt'), 'Block optimizer list not found.'
        # default options for now
        block_sch = [
            ReduceLROnPlateau(
                b_opt
            ) for b_opt in self.block_opt
        ]
        output_sch = ReduceLROnPlateau(
            self.output_opt
        ) if self.num_blocks > 1 else None
        return output_sch, block_sch

    def configure_optimizers(self):
        self.output_opt, self.block_opt = self.get_optimizers()

    def configure_schedulers(self):
        self.output_sch, self.block_sch = self.get_schedulers()

    def get_checkpoint(self):
        pass

    def load_from_checkpoint(self,checkpoint):
        # create model from checkpoint file
        pass

class RAE_block(nn.Module):

    def __init__(self, input_size, encoder_size, decoder_size, dropout, 
                num_layers_encoder=1, bidirectional_encoder=True, 
                decoder_ic_prior_params={}, rand_samp=True):
        super().__init__()

        self.input_size = input_size
        self.encoder_size = encoder_size
        self.decoder_size = decoder_size
        self.dropout = dropout
        self.rand_samp  = rand_samp

        # initalize model blocks
        self.encoder    = Encoder(
            input_size=input_size,
            output_size=decoder_size,
            hidden_size=encoder_size,
            dropout=dropout,
            num_layers=num_layers_encoder,
            bidirectional=bidirectional_encoder)
        self.decoder    = Decoder(
            hidden_size=decoder_size,
            dropout=dropout,
            num_layers=1,
            bidirectional=False
        )
        self.block_out  = nn.Linear(in_features=decoder_size,out_features=input_size)

        # create variational prior parameters for decoder IC
        self.dec_ic_prior_params = {
            'mean': decoder_ic_prior_params.get('mean',0.),
            'logvar': decoder_ic_prior_params.get('logvar',0.),
        }

    def forward(self, input):
        # forward pass
        batch_size, seq_len, input_size = input.size()
        assert input_size == self.input_size, f"Input tensor size {input_size} must match model input_size {self.input_size}"
        mean, logvar = self.encoder(input)
        kl_div = self.compute_dec_ic_kl_div(mean, logvar)
        decoder_ic = self.sample_decoder_ic(mean, logvar) # model latent state
        decoder_input = self.decoder.gen_input(batch_size,seq_len)
        decoder_out = self.decoder(decoder_input,decoder_ic)
        return self.block_out(decoder_out), decoder_out, decoder_ic, kl_div

    def compute_dec_ic_kl_div(self,dec_ic_posterior_mean,dec_ic_posterior_logvar):
        kl_div = kl_div_normals(
            self.dec_ic_prior_params['mean'],
            dec_ic_posterior_mean,
            self.dec_ic_prior_params['logvar'],
            dec_ic_posterior_logvar,
        ).mean(dim=0).sum()
        return kl_div # average this? sum this?

    def compute_decoder_l2(self):
        return self.decoder.rnn.hidden_weight_l2_norm()

    def sample_decoder_ic(self, mean, logvar):
        if self.rand_samp:
            return sample_gaussian(mean, logvar)
        else:
            return mean

class Encoder(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, dropout, 
                num_layers=1, bidirectional=True):
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

    def forward(self, input):
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

    def forward(self, input, h0):
        gen_out = self.rnn(input,h0=h0)
        gen_out = self.dropout_layer(gen_out)
        return gen_out

    def gen_input(self,batch_size,seq_len):
        return torch.zeros(batch_size,seq_len,self.input_size) #TODO: make this force to self.device

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

def kl_div_normals(mean_1, mean_2, logvar_1, logvar_2):
    """
    kl_div_normals

    compute the kl divergence of two normal distributions, N_2 from N_1

    Args:
        mean_1 (float): mean of N_1
        mean_2 (float): mean of N_2
        logvar_1 (float): log variance of N_1
        logvar_2 (float): log_variance of N_2

    Returns:
        _type_: _description_
    """
    return 0.5 * (logvar_2 - logvar_1 + torch.exp(logvar_1 - logvar_2) \
        + (mean_1 - mean_2).pow(2)/torch.exp(logvar_2) + 1)
