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


HELP = "distill.py -m"


if __name__ == '__main__':

    set_seed(1)
    cfg = parse(Config)
    pprint(cfg)

    start_time = time.time()
    print("加载数据...")

    train_text, train_label = get_dataset(cfg.train_path)
    test_text, test_label = get_dataset(cfg.test_path)
    train_loader = get_loader(train_text, train_label, cfg.tokenizer)
    test_loader = get_loader(test_text, test_label, cfg.tokenizer)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    T_model = BERT_Model(cfg).to(cfg.device)

    if cfg.train_teacher:
        teacher_train(T_model, cfg, train_loader, test_loader)

    if cfg.train_student:
        S_model = biLSTM(cfg).to(cfg.device)
        student_train(T_model, S_model, cfg, train_loader, test_loader)

