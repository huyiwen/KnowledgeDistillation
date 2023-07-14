# -*- coding: utf-8 -*-
"""
The Bidirectional LSTM model wrapper for non-MPO version and MPO version.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from compress_tools.MPOtorch import LinearDecomMPO, EmbeddingMPO
from torch import Tensor
from typing import Optional, Tuple, List


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, config):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = config.LSTM_bias
        self.peephole = config.LSTM_peephole
        self.use_mpo = config.use_mpo and "lstm" in config.mpo_type
        if self.use_mpo:
            self.xh_mpo_config = config.xh_mpo
            self.hh_mpo_config = config.hh_mpo
            # self.xh = nn.Linear(input_size, hidden_size * 4, bias=self.bias)
            # self.hh = nn.Linear(hidden_size, hidden_size * 4, bias=self.bias)
            self.xh = LinearDecomMPO(input_size, hidden_size*4, *self.xh_mpo_config, bias=self.bias)
            self.hh = LinearDecomMPO(hidden_size, hidden_size*4, *self.hh_mpo_config, bias=self.bias)
        else:
            self.xh = nn.Linear(input_size, hidden_size * 4, bias=self.bias)  #
            self.hh = nn.Linear(hidden_size, hidden_size * 4, bias=self.bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input: Tensor, hx: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Inputs:
              input: of shape (batch_size, input_size)
              hx: of shape (batch_size, hidden_size)
        Outputs:
              hy: of shape (batch_size, hidden_size)
              cy: of shape (batch_size, hidden_size)
        """

        hx, cx = hx

        if self.peephole:

            xh_input = self.xh(input)
            gates = xh_input + self.hh(hx)
            input_gate, forget_gate, _, output_gate = gates.chunk(4, dim=1)

            i_t = torch.sigmoid(input_gate)
            f_t = torch.sigmoid(forget_gate)
            o_t = torch.sigmoid(output_gate)

            cy = cx * f_t + i_t * torch.sigmoid(xh_input)[:, 2*self.hidden_size:3*self.hidden_size]
            hy = torch.tanh(o_t * cy)

        else:
            # print(input.shape)
            gates = self.xh(input) + \
                self.hh(hx)
            input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, dim=1)

            i_t = torch.sigmoid(input_gate)
            f_t = torch.sigmoid(forget_gate)
            g_t = torch.tanh(cell_gate)
            o_t = torch.sigmoid(output_gate)

            cy = cx * f_t + i_t * g_t
            hy = o_t * torch.tanh(cy)

        return (hy, cy)



class biLSTMMPO(nn.Module):
    def __init__(self, input_size, hidden_size, config, num_layers=1, dropout=0, bidirectional=True, bias=True, batch_first=True):
        super().__init__()

        if dropout != 0 or bidirectional is False or batch_first is False:
            raise NotImplementedError("dropout is not implemented in LSTM")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn_cell_list = nn.ModuleList()
        self.rnn_cell_list.append(LSTMCell(self.input_size, self.hidden_size, config))
        for _ in range(1, self.num_layers):
            self.rnn_cell_list.append(LSTMCell(self.hidden_size, self.hidden_size, config))


    def forward(self, input, init_states: Optional[Tuple[List[Tuple[Tensor]]]] = None):
        """Assumes x is of shape (batch, sequence, feature)"""

        # init_states is a tuple of two tensors (h, c)
        max_batch_size, seq_sz, _ = input.size()
        if init_states is None:
            h_c = torch.zeros(max_batch_size, self.hidden_size,
                              dtype=input.dtype, device=input.device)
            h_c = [[None, None]] + [[h_c, h_c] for _ in range(self.num_layers)]
            init_states = (h_c, h_c)

        # h shape: Layer * 2 * [Seq Bat Hid]
        h0_t, hT_t = init_states

        # LSTM
        # out shape: [Seq Bat Hid]
        out0 = list()
        outT = list()
        for t in range(seq_sz):
            # initial input as layer 0
            h0_t[0][0] = input[:, t, :]
            hT_t[0][0] = input[:, -(t + 1), :]
            for layer in range(1, self.num_layers+1):
                # LSTM Cell forward
                h0_t[layer] = self.rnn_cell_list[layer-1](h0_t[layer - 1][0], h0_t[layer])
                hT_t[layer] = self.rnn_cell_list[layer-1](hT_t[layer - 1][0], hT_t[layer])
            # save the hidden states of the last layer
            out0.append(h0_t[self.num_layers][0].unsqueeze(0))
            outT.append(hT_t[self.num_layers][0].unsqueeze(0))

        hidden_seq = torch.cat([torch.cat(out0), torch.cat(outT)], dim=2)

        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()

        return hidden_seq, (h0_t, hT_t)



class biLSTM(nn.Module):

    def __init__(self, config):
        super(biLSTM, self).__init__()
        self.use_mpo = config.use_mpo
        if self.use_mpo and "embedding" in config.mpo_type:
            self.embedding_mpo_config = config.embedding_mpo
            self.Embedding = EmbeddingMPO(21128, 300, *self.embedding_mpo_config)
        else:
            self.Embedding = nn.Embedding(21128, 300)

        if self.use_mpo and "lstm" in config.mpo_type:
            self.lstm = biLSTMMPO(
                input_size=300,
                hidden_size=300,
                num_layers=1,
                config=config,
            )
        else:
            self.lstm = nn.LSTM(input_size=300, hidden_size=300,
                                  num_layers=1, batch_first=True,
                                  dropout=0, bidirectional=True)

        if self.use_mpo and "fc" in config.mpo_type:
            self.fc1_mpo_config = config.fc1_mpo
            # print("fc1_mpo_config: ", self.fc1_mpo_config)
            self.fc1 = LinearDecomMPO(300*2, 192, *self.fc1_mpo_config)

            self.fc2_mpo_config = config.fc2_mpo
            self.fc2 = LinearDecomMPO(192, config.num_classes, *self.fc2_mpo_config)
        else:
            self.fc1 = nn.Linear(300*2, 192)
            self.fc2 = nn.Linear(192, config.num_classes)

    def forward(self, x, hidden=None):
        if self.use_mpo:
            x = self.Embedding(x)
            lstm_out, hidden = self.lstm(x, hidden)
            out = self.fc1(lstm_out)
            activated_t = F.relu(out)
            linear_out = self.fc2(activated_t)
            linear_out = torch.max(linear_out, dim=1)[0]
        else:
            x = self.Embedding(x)
            lstm_out, hidden = self.lstm(x, hidden)
            out = self.fc1(lstm_out)
            activated_t = F.relu(out)
            linear_out = self.fc2(activated_t)
            linear_out = torch.max(linear_out, dim=1)[0]
            # print("linear_out.shape: ", linear_out.shape)
        return linear_out
