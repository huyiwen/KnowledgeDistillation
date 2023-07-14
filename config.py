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

    class_list: List[str] = arg(default=map(lambda x: x.strip(), open('data/class_multi1.txt').readlines()))
    train_path: str = 'data/train.json'
    test_path: str = 'data/test.json'
    teacher_save_path: str = 'saved_dict/teacher.ckpt'        # 模型训练结果
    student_save_path: str = 'saved_dict/student.ckpt'        # 模型训练结果

    device: torch.DeviceObjType = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

    train_teacher: int = 0
    train_student: int = 1
    require_improvement: int = 1000                           # 若超过1000batch效果还没提升，则提前结束训练
    num_classes: int = arg(default=len(open('data/class_multi1.txt').readlines()))
    teacher_num_epochs: int = 3                               # epoch数
    student_num_epochs: int = 3                               # epoch数

    batch_size: int = 64                                      # 128mini-batch大小
    pad_size: int = 32                                        # 每句话处理成的长度(短填长切)
    learning_rate: float = 5e-4                                 # 学习率
    bert_path: str = './bert_pretrain'
    tokenizer: BertTokenizer = BertTokenizer.from_pretrained(bert_path)
    hidden_size: int = 768

    LSTM_bias: bool = True
    LSTM_peephole: bool = False

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

