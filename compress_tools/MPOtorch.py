# -*- coding: utf-8 -*-
import numpy as np
from torch import nn as nn
import torch
import logging
import os
from compress_tools.Matrix2MPO import MPO
from torch.nn import functional as F
from torch import nn


os.environ['KMP_DUPLICATE_LIB_OK']='True'
logger = logging.getLogger(__name__)


def linear_act(x):
    return x


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.contiguous().view((-1,)+self.shape[0])


class TransposeLayer(nn.Module):
    def __init__(self, *args):
        super(TransposeLayer, self).__init__()
    def forward(self, x):
        return torch.transpose(x,0,1)


class LinearDecomMPO(nn.Module):
    '''
    compress using MPO method
    ref: Compressing deep neural networks by matrix product operators
    '''
    def __init__(
        self,
        input_size,
        output_size,
        mpo_input_shape,
        mpo_output_shape,
        trunc_num,
        bias = True,
        tensor_set=None,
        bias_tensor=None,
        device=None,
    ):
        super(LinearDecomMPO, self).__init__()
        self.trunc_num = trunc_num
        self.mpo_input_shape = np.array(mpo_input_shape)
        self.mpo_output_shape = np.array(mpo_output_shape)
        self.tensor_set = None
        self.mpo = MPO(self.mpo_input_shape, self.mpo_output_shape, self.trunc_num)

        if tensor_set:
            self._from_pretrained(tensor_set, bias_tensor, bias, device)
        else:
            # not register as parameter
            _linear = nn.Linear(input_size, output_size, bias=bias, device=device)
            mpo_tensor_set, _, _ = self.mpo.matrix2mpo(_linear.weight.data.cpu().numpy())
            self._from_pretrained(
                tensor_set=mpo_tensor_set,
                bias_tensor=_linear.bias if bias else None,
                bias=bias,
                device=device
            )
            # print(self.get_weight().shape, _linear.weight.data.shape, (input_size, output_size))

    def get_weight(self):
        return self.mpo.mpo2matrix(self.tensor_set)

    def forward(self, x):
        ##################### use rebulid
        res = x.reshape(-1, x.shape[-1])
        res = F.linear(res, self.get_weight(), self.bias_tensor)
        ##################### use rebuild
        ori_shape=x.shape

        return res.view((tuple(ori_shape[:-1])+(-1,)))
        # return res

    def _from_pretrained(self, tensor_set, bias_tensor=None, bias=True, device=None):
        # register tensor_set as parameter
        if device is not None:
            self.tensor_set = torch.nn.ParameterList([nn.Parameter(torch.from_numpy(i).to(device)) for i in tensor_set])
        else:
            self.tensor_set = torch.nn.ParameterList([nn.Parameter(torch.from_numpy(i)) for i in tensor_set])

        if bias:
            self.bias_tensor = bias_tensor
        else:
            logger.info("Check no bias")
            self.bias_tensor = None


class EmbeddingMPO(nn.Module):
    '''
    use MPO decompose word embedding
    '''
    def __init__(self, num_embeddings, embedding_dim, mpo_input_shape, mpo_output_shape, truncate_num, tensor_set, device=None, **kwargs):
        super(EmbeddingMPO, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.mpo = MPO(mpo_input_shape, mpo_output_shape, truncate_num)
        self.tensor_set = None
        if tensor_set is not None:
            self._from_pretrained(tensor_set)
        else:
            _embedding = nn.Embedding(num_embeddings, embedding_dim)
            self._from_pretrained(_embedding.weight.data.cpu().numpy(), device=device)

    def forward(self, input):
        weight_rebuild = self.mpo.mpo2matrix(self.tensor_set)[:self.num_embeddings]
        return F.embedding(input, weight_rebuild)

    def _from_pretrained(self, tensor_set, device=None):
        # register tensor_set as parameter
        if device is not None:
            self.tensor_set = torch.nn.ParameterList([nn.Parameter(torch.from_numpy(i).to(device)) for i in tensor_set])
        else:
            self.tensor_set = torch.nn.ParameterList([nn.Parameter(torch.from_numpy(i)) for i in tensor_set])
