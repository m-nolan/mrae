# custom RNN implementations, i.e. GRU with "brick on the gas" z-gate bias

import torch
import torch.nn as nn

class GRU_Cell_Modified(nn.Module):

    def __init__(self, input_size, hidden_size, update_bias=1.0):
        super().__init__()
        self.input_size     = input_size
        self.hidden_size    = hidden_size
        self.update_bias    = update_bias

        # concat sizes for collected hidden states
        self._xh_size   = input_size + hidden_size
        self._rz_size   = hidden_size * 2

        # r, z <- W([x, h]) + b (reset gate r, update gate z)
        self.fc_xh_rz = nn.Linear(
            in_features=self._xh_size,
            out_features=self._rz_size
        )

        # hhat <- W([x, h*r]) + b (candidate hidden state hhat)
        self.fc_xhr_hhat = nn.Linear(
            in_features=self._xh_size,
            out_features=self.hidden_size
        )

    def forward(self, x, h):
        """Modified GRU cell forward pass

        Args:
            x (torch.Tensor): GRU input tensor. Last dimension must be size self.input_size
            h (torch.Tensor): GRU hidden state tensor. Last dimension must be size self.hidden_size

        Returns:
            h_new (torch.Tensor): Updated GRU hidden state tensor. Size self.hidden_size
        """

        xh = torch.cat([x,h], dim=1)

        r, z = torch.split(
            self.fc_xh_rz(xh),
            split_size_or_sections=self.hidden_size,
            dim = 1
        )
        r = torch.sigmoid(r)
        z = torch.sigmoid(z + self.update_bias)

        xrh = torch.cat([x, torch.mul(r, h)], dim=1)
        hhat = torch.tanh(self.fc_xhr_hhat(xrh))

        return torch.mul(z, h) + torch.mul(1-z, hhat)