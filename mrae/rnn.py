# custom RNN implementations, i.e. GRU with "brick on the gas" z-gate bias

import torch
import torch.nn as nn

class GRU_Modified(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size     = input_size
        self.hidden_size    = hidden_size

        self.gru_cell   = GRU_Cell_Modified(
            input_size=input_size,
            hidden_size=hidden_size
        )

    def forward(self, input, h0=None):
        """
        Modified GRU forward pass

        Args:
            input (torch.Tensor): Input sequence tensor. Must be size [*,n_sample,input_size]
            h0 (torch.Tensor, optional): Hidden state initial condition tensor. Must be size [*,hidden_size]. Will initialize with zero tensor if input value is None. Defaults to None.

        Returns:
            output (torch.Tensor): Output sequence tensor size [*, n_sample, input_size]
        """
        batch_size, n_samples, input_size = input.size()
        assert input_size == self.input_size, "Input tensor size mismatch"
        output = torch.empty(batch_size,n_samples,self.hidden_size).to(self.gru_cell.fc_xh_rz.weight.device)
        if h0 is None:
            h0 = torch.zeros(batch_size,self.hidden_size).to(self.gru_cell.fc_xh_rz.weight.device)
        h_in = h0
        for s_idx in range(n_samples):
            h_out = self.gru_cell(input[:,s_idx,:],h_in)
            output[:,s_idx,:] = h_out
            h_in = h_out
        return output

    def hidden_weight_l2_norm(self):
        return self.gru_cell.hidden_weight_l2_norm()

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

    def hidden_weight_l2_norm(self):
        return self.fc_xh_rz.weight.norm(2).pow(2) / self.fc_xh_rz.weight.numel() + \
            self.fc_xhr_hhat.weight.norm(2).pow(2) / self.fc_xhr_hhat.weight.numel()
