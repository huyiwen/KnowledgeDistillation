# -*- coding: utf-8 -*-
"""
# @Time    : 2020/5/19 下午3:52
# @Author  : HOY
# @Email   : huangouyan@changingedu.com
# @File    : utils.py
# @Software: PyCharm
"""
import re
import time
import json
import torch
import numpy as np
from datetime import timedelta
from torch.utils.data import DataLoader, TensorDataset
from transformers import PreTrainedTokenizer
from prefetch_generator import BackgroundGenerator

class DataLoaderX(torch.utils.data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def get_chinese(text):
    text = ''.join(re.findall(r'[A-Za-z0-9\u4e00-\u9fa5]', text))
    return text


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def set_seed(n):
    np.random.seed(n)
    torch.manual_seed(n)
    torch.cuda.manual_seed_all(n)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样


def get_dataset(path):
    with open(path, 'r') as f_train:
        data_pair = json.load(f_train)
    text = [get_chinese(data['text']) for data in data_pair]
    label = [(data['label']) for data in data_pair]
    return text, label


def get_loader(x, y, tokenizer: PreTrainedTokenizer, max_seq_length, batch_size):
    data = tokenizer.batch_encode_plus(x, max_length=max_seq_length, padding='max_length', truncation='longest_first')
    input_ids = data['input_ids']
    # print(input_ids[:10])
    mask = data['attention_mask']
    data_loader = DataLoaderX(
        TensorDataset(
            torch.LongTensor(input_ids),
            torch.LongTensor(mask),
            torch.LongTensor(y)
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    return data_loader




