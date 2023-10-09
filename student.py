# -*- coding: utf-8 -*-
"""
# @Time    : 2020/5/19 上午10:53
# @Author  : HOY
# @Email   : huangouyan@changingedu.com
# @File    : student.py
# @Software: PyCharm
"""
import time
from typing import Tuple, Union, NoReturn, Optional

import numpy as np
import torch
from sklearn import metrics
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.nn import ParameterList, Module
from einops import rearrange
import wandb

from teacher import teacher_predict, teacher_load
from utils import get_time_dif
from config import Config


# 损失函数
def get_loss(t_output: Tensor, s_output: Tensor, label: Tensor, a: float, T: float, loss_align: bool, loss_func: str = "CosineEmbeddingLoss", loss_weight: Optional[float] = None, epoch=-1) -> Tensor:

    a, T = a / (a + T), T / (a + T)

    loss1 = nn.CrossEntropyLoss()
    loss2 = nn.MSELoss()
    if loss_func == 'CosineEmbeddingLoss':
        _loss3 = nn.CosineEmbeddingLoss()
        def loss3(s_tensor: Tensor, t_tensor: Tensor) -> Tensor:
            s_tensor = rearrange(s_tensor, 'b i j d -> (b i j d)')
            t_tensor = rearrange(t_tensor, 'b i j d -> (b i j d)')
            # print(s_tensor.shape)
            return _loss3(s_tensor, t_tensor, torch.ones(()))
    else:
        loss3 = getattr(nn, loss_func)()

    if loss_align:

        # 余弦相似度约束
        s_logits = s_output[0]
        t_logits = t_output[0]
        s_tensorsets: Tuple[ParameterList] = s_output[1]
        t_tensorsets: Tuple[ParameterList] = t_output[1]

        loss = a * loss1(s_logits, label) + T * loss2(t_logits, s_logits)

        for s_tensorset, t_tensorset in zip(s_tensorsets, t_tensorsets):
            s_size = len(s_tensorset)
            for i in range(s_size // 2):
                s_tensor = s_tensorset[i]
                t_tensor = t_tensorset[i]
                if s_tensor.size() == t_tensor.size():
                    # 余弦相似度约束
                    loss += loss_weight * loss3(s_tensor, t_tensor)

                s_tensor = s_tensorset[s_size - i - 1]
                t_tensor = t_tensorset[s_size - i - 1]
                if s_tensor.size() == t_tensor.size():
                    loss += loss_weight * loss3(s_tensor, t_tensor)
        # print(loss1(s_logits, label),loss2(t_logits, s_logits))

        return loss

    else:

        # 无约束
        s_logits = s_output
        t_logits = t_output

        base_loss = loss1(s_logits, label)  # CrossEntropy，不需要softmax
        distillation_loss = loss2(torch.softmax(t_logits, dim=-1), torch.softmax(s_logits, dim=-1))  # MSELoss

        if epoch % 50 == 0:
            print("s_logits", s_logits, "label", label)
            print("t_logits", t_logits)
            print("base_loss", base_loss, "distillation_loss", distillation_loss)
        loss = a * base_loss + T * distillation_loss

        return loss


def student_train(T_model: Module, S_model: Module, config: Config, train_loader: DataLoader, test_loader: DataLoader) -> NoReturn:
    # T_model = T_model.train()
    T_model = teacher_load(T_model, config)
    t_train_outputs = teacher_predict(T_model, config, train_loader)
    t_test_outputs = teacher_predict(T_model, config, test_loader)
    # loss, acc = student_evaluate(T_model, config, t_test_outputs, test_loader)
    # print("teacher loss", loss, "teacher acc", acc)
    total_params = sum(p.numel() for p in S_model.parameters())
    print(f'{total_params:,} total parameters.')

    # 分别设置lr
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in S_model.named_parameters() if not "tensor_set" in n],
            "lr": config.distill_lr,
        },
        {
            "params": [p for n, p in S_model.named_parameters() if "tensor_set" in n],
            "lr": config.mpo_lr,
        },
    ]
    optimizer = getattr(torch.optim, config.distill_optimizer)(optimizer_grouped_parameters, lr=config.distill_lr)
    print("distill_lr", config.distill_lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.student_num_epochs * len(train_loader))

    total_batch = 0
    tra_best_loss = float('inf')
    dev_best_loss = float('inf')

    start_time = time.time()
    for epoch in range(config.student_num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.student_num_epochs))
        for i, (texts, _, label) in enumerate(train_loader):
            S_model.train()
            # print(texts, label)
            texts = texts.to(config.device)
            label = label.to(config.device)
            optimizer.zero_grad()
            s_logits = S_model(texts)
            # print(t_train_logits[i].shape, s_logits.shape, label.shape)

            # loss = nn.CrossEntropyLoss()(s_outputs, label.long())
            # TODO 设置a=0,T=1时无法学习（二分类正确率50%）？
            loss = get_loss(
                t_output=t_train_outputs[i],
                s_output=s_outputs,
                label=label.long(),
                a=1,
                T=1,
                loss_align=config.loss_align,
                loss_func=config.loss_func,
                loss_weight=config.loss_weight,
                epoch=total_batch,
            )

            loss.backward()
            optimizer.step()
            scheduler.step()
            if total_batch % 50 == 0:

                # print(texts[0])
                # for name, param in S_model.named_parameters():
                #     print("==> name", name, param.grad.shape, param.grad)
                # 计算准确率
                cur_pred = torch.squeeze(s_outputs[0] if config.loss_align else s_outputs, dim=1)
                train_acc = metrics.accuracy_score(
                    label.cpu().long(),
                    torch.max(cur_pred, dim=1)[1].cpu().numpy()
                )

                # print(list(S_model.named_parameters()))
                # 评测在验证集上的效果
                dev_loss, dev_acc = student_evaluate(S_model, config, t_test_outputs, test_loader)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(S_model.state_dict(), config.student_save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6},  LR: {7:.2f}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve, scheduler.get_last_lr()[0]))
                wandb.log({"train_loss": loss.item(), "train_acc": train_acc, "loss": dev_loss, "acc": dev_acc, 'epoch': epoch, "lr": scheduler.get_last_lr()[0]})

            total_batch += 1

    loss, acc = student_evaluate(S_model, config, t_test_outputs, test_loader)
    print("loss", loss, "acc", acc)


def student_evaluate(S_model, config, t_logits, test_loader):
    S_model.eval()
    predict_all = []
    labels_all = []
    loss_total = 0
    with torch.no_grad():
        for i, (texts, _, label) in enumerate(test_loader):
            texts = texts.to(config.device)
            label = label.to(config.device)
            s_outputs = S_model(texts)

            loss = get_loss(t_outputs[i], s_outputs, label.long(), 0, 2, config.loss_align, config.loss_func, config.loss_weight)

            loss_total += loss.item()

            cur_pred = torch.squeeze(s_outputs[0] if config.loss_align else s_outputs, dim=1)
            predic = torch.max(cur_pred, dim=1)[1].cpu().numpy()
            label = label.data.cpu().numpy()
            labels_all = np.append(labels_all, label)
            predict_all = np.append(predict_all, predic)
    acc = metrics.accuracy_score(labels_all, predict_all)
    return loss_total/len(test_loader), acc



