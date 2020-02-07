from typing import Callable, Generic, TypeVar, Iterable, Dict, Tuple
import torch
from torch import nn, Tensor, optim
from framework.Loss import Loss
from framework.module import NamedModule
from framework.segmentation.Mask import Mask
from itertools import chain

T1 = TypeVar('T1')
T2 = TypeVar('T2')


class CycleGAN(Generic[T1, T2]):
    def __init__(self,
                 g_forward: NamedModule,
                 g_backward: NamedModule,
                 loss_1: Dict[str, Callable[[T1, T1], Loss]],
                 loss_2: Dict[str, Callable[[T2, T2], Loss]],
                 lr: float = 0.0001,
                 betas=(0.5, 0.999)):

        self.g_forward = g_forward
        self.g_backward = g_backward
        self.loss_1 = loss_1
        self.loss_2 = loss_2

        self.opt = optim.Adam(
            chain(
              g_forward.parameters(),
              g_backward.parameters()
            ),
            lr=lr,
            betas=betas)

    def loss_forward(self, condition: Dict[str, T1]) -> Loss:

        t2 = self.g_forward(condition)
        condition_pred: Dict[str, T1] = self.g_backward(t2)

        loss = Loss.ZERO()
        for name in condition.keys():
            loss += self.loss_1[name](condition_pred[name], condition[name])

        return loss

    def loss_backward(self, condition: Dict[str, T2]) -> Loss:
        condition_pred: Dict[str, T2] = self.g_forward(self.g_backward(condition))

        loss = Loss.ZERO()
        for name in condition.keys():
            loss += self.loss_2[name](condition_pred[name], condition[name])

        return loss

    def train(self, t1: Dict[str, T1], t2: Dict[str, T2]):

        self.g_forward.zero_grad()
        self.g_backward.zero_grad()
        self.loss_forward(t1).minimize()
        self.opt.step()

        self.g_forward.zero_grad()
        self.g_backward.zero_grad()
        self.loss_backward(t2).minimize()
        self.opt.step()
