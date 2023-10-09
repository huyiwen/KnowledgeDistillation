# -*- coding: utf-8 -*-
"""
# @Time    : 2021/10/24 9:40 下午
# @Author  : HOY
# @Email   : 893422529@qq.com
# @File    : train_main.py
"""
from utils import get_dataset, get_loader, get_time_dif, set_seed
from config import Config
from student import student_train
from teacher import teacher_train
from models.bert import BERT_Model
from models.lstm_mpo import biLSTM
import time
from datargs import parse
from pprint import pprint
import datasets
import numpy as np
import torch
from datetime import datetime
import wandb


HELP = "distill.py -m"


if __name__ == '__main__':

    cfg = parse(Config)
    set_seed(cfg.seed)
    if cfg.class_list is None:
        cfg.class_list = map(lambda x: x.strip(), open('data/class_multi1.txt').readlines())
    if cfg.num_classes is None:
        cfg.num_classes = len(open('data/class_multi1.txt').readlines())
    pprint(cfg)
    name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    wandb.init(
        project="bilstm",
        name=name,
        config=cfg,
        group=name,
    )

    start_time = time.time()
    print("加载数据...")

    data = datasets.load_from_disk(cfg.data)

    train_text, train_label = data['train']['text'], data['train']['label']
    test_text, test_label = data['test']['text'], data['test']['label']

    train_loader = get_loader(train_text, train_label, cfg.tokenizer, cfg.max_seq_length, cfg.finetune_batch_size)
    test_loader = get_loader(test_text, test_label, cfg.tokenizer, cfg.max_seq_length, cfg.distill_batch_size)
    print(next(iter(train_loader))[0])
    cfg.tokenizer.decode(next(iter(train_loader))[0][0])

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    T_model = BERT_Model(cfg).to(cfg.device)
    print(T_model)

    if cfg.train_teacher:
        teacher_train(T_model, cfg, train_loader, test_loader)

    if cfg.train_student:
        # embed = T_model.state_dict()['bert.embeddings.word_embeddings.weight'].clone().detach()
        print(cfg.device)
        S_model = biLSTM(cfg).to(cfg.device)
        print(S_model)
        student_train(T_model, S_model, cfg, train_loader, test_loader)

