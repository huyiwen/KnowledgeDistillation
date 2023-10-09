# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from config import Config
from models.linear_mpo_ import LinearDecomMPO


class BERT_Model(nn.Module):

    def __init__(self, config):
        super(BERT_Model, self).__init__()
        self.bert = AutoModel.from_pretrained(config.bert_path)
        for param in list(self.bert.parameters())[:-4]:
            param.requires_grad = False

        self.use_mpo = config.use_mpo and config.loss_align
        if self.use_mpo:
            self.tfc_mpo_config = config.tfc_mpo
            self.fc = LinearDecomMPO(config.bert_hidden_size, 192, *self.tfc_mpo_config)
            self.tfc1_mpo_config = config.tfc1_mpo
            self.fc1 = LinearDecomMPO(192, config.num_classes, *self.tfc1_mpo_config)
        else:
            self.fc = nn.Linear(config.bert_hidden_size, 192)
            self.fc1 = nn.Linear(192, config.num_classes)

    def forward(self, context, mask):
        # print("context: ", context.shape)  # [batch_size, seq_len]
        out = self.bert(context, attention_mask=mask)
        out = self.fc(out[1])
        out = F.relu(out)
        out = self.fc1(out)
        return out
