# nn.py
# author: cyn
# email: yunachen@stu.xmu.edu.cn
# coding=utf-8

import torch


def maxout(input):
   input = input.view(input.shape[0], input.shape[1], -1, 2)
   values, indices = torch.max(input, -1)
   return values









