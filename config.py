# -*- coding: utf-8 -*-
"""
# @Time    : 2021/11/7 8:30 下午
# @Author  : HOY
# @Email   : 893422529@qq.com
# @File    : config.py
"""
import torch
from transformers import BertTokenizer
from dataclasses import dataclass, field
from typing import List
from datargs import arg


@dataclass
class Config(object):

    class_list: List[str] = ('0', '1')
    teacher_save_path: str = 'saved_dict/teacher.ckpt'
    student_save_path: str = 'saved_dict/student.ckpt'
    data: str = "/home/huyiwen/datasets/sst2"
    seed: int = 42

    device: torch.DeviceObjType = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_teacher: int = 0
    train_student: int = 1
    require_improvement: int = 1000
    num_classes: int = 2
    teacher_num_epochs: int = 3
    student_num_epochs: int = 3
    finetune_optimizer: str = 'AdamW'
    distill_optimizer: str = arg(default='AdamW', aliases=['--opt'])

    finetune_batch_size: int = 64
    distill_batch_size: int = 64
    max_seq_length: int = 128
    finetune_lr: float = 5e-4
    distill_lr: float = arg(default=0.001, aliases=['--lr'])
    mpo_lr: float = 0.0002
    bert_path: str = '/home/huyiwen/pretrained/bert-base-uncased-SST-2'
    tokenizer: BertTokenizer = BertTokenizer.from_pretrained(bert_path)
    bert_hidden_size: int = 768

    LSTM_embedding_dim: int = 300
    LSTM_hidden_dim: int = 300
    LSTM_bias: bool = True
    LSTM_peephole: bool = False
    FC_dim: int = 192

    use_mpo: bool = False
    mpo_type: List[str] = arg(default=("fc", "lstm", "embedding"))
    truncate_num: int = 10000

    embedding_input_shape: List[int] = arg(default=list)  # 21128
    embedding_output_shape: List[int] = arg(default=list)  # 300

    fc1_input_shape: List[int] = arg(default=())  # 600
    fc1_output_shape: List[int] = arg(default=())  # 192

    fc2_input_shape: List[int] = arg(default=())  # 192
    fc2_output_shape: List[int] = arg(default=())  # 5 (Tnews)

    xh_input_shape: List[int] = arg(default=())  # 300
    xh_output_shape: List[int] = arg(default=())  # 1200

    hh_input_shape: List[int] = arg(default=())  # 300
    hh_output_shape: List[int] = arg(default=())  # 1200


    def __getattr__(self, name):
        if name.endswith("_mpo"):
            return (
                getattr(self, name[:-4] + "_input_shape"),
                getattr(self, name[:-4] + "_output_shape"),
                self.truncate_num,
            )
        else:
            raise AttributeError(f"{__class__.__name__} has no attribute {name}")

