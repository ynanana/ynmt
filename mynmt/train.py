# train.py
# author: cyn
# email: yunachen@stu.xmu.edu.cn
# coding=utf-8

import torch
import torch.nn.utils as utils
import torch.nn as nn

import sys
import os
import math
import re
from nltk.translate.bleu_score import corpus_bleu

from model import cgru
from utils.data import *
from utils.config import *
from utils.nns import *


# path
strain_path = args.train_data[0]
ttrain_path = args.train_data[1]
svoc_path = args.vocabulary[0]
tvoc_path = args.vocabulary[1]
sdev_path = args.dev_data[0]


def train(src_voc, tgt_voc, model, loss_func, optimizer):
    dev_data = eval_iterator(sdev_path, src_voc)
    global_step = 0
    running_loss = 0.0
    best_bleu = 0.0
    for epoch in range(100):
        train_data = train_iterator(strain_path, ttrain_path, args.batch_size,
                                    src_voc, tgt_voc)
        for step, data in enumerate(train_data):
            global_step += 1
            x, y, x_len, y_len = data
            loss = train_batch(data, model, loss_func, optimizer)
            running_loss += float(loss)
            logging.info("epoch:" + str(epoch) + ' step:' + str(global_step)
                         + " x:" + str(x.shape) + ' y:' + str(y.shape)
                         + " loss:" + str(loss.item()) + " lr:"
                         + str(args.learning_rate))
            if global_step != 0 and global_step % 5000 == 0:
                logging.info("epoch:" + str(epoch) + " step loss:"
                             + str(global_step) + ' '
                             + str(running_loss / 5000))
                running_loss = 0.0
                with torch.no_grad():
                    bleu = evaluate(dev_data, model)
                    logging.info("step:" + str(global_step)
                                 + " bleu:" + str(bleu * 100))
                if bleu > best_bleu:
                    best_bleu = bleu
                    torch.save(model.state_dict(), args.checkpoint)
        if args.model == "cgru":
            args.learning_rate = args.learning_rate * 0.5
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.learning_rate



def train_batch(data, model, loss_func, optimizer):
    model.train(True)

    x, y, x_len, y_len = data
    x = x.cuda()
    y = y.cuda()
    x_len = x_len.cuda()
    y_len = y_len.cuda()

    output = model.forward(x, y, x_len, y_len)
    y = y.view(-1)
    output = output.view(-1, tvocab_size)

    loss = loss_func(output, y)
    optimizer.zero_grad()

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()

    return loss


def evaluate(data, model):
    model.eval()
    fin = open(args.dev_data[1], 'r')
    reference = []
    for line in fin.readlines():
        reference.append([line.strip().split(' ')])

    candidate = []
    for i, (x, x_len) in enumerate(data):
        logging.info(i)
        x = x.cuda()
        x_len = x_len.cuda()

        output = model.evaluate(x, x_len)
        sent = []
        for id in output:
            if id == EOS:
                break
            sent.append(tgt_voc.id2word[id])

        sent = ' '.join(sent)
        bpe = re.compile('(@@ )|(@@ ?$)')
        sent = bpe.sub('', sent)
        sent = sent.split()

        candidate.append(sent)

    bleu = corpus_bleu(reference, candidate)
    return bleu


def initial_params(model, mode="uniform"):
    if mode == "uniform":
        for param in model.parameters():
            param.data.uniform_(-0.08, 0.08)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        logging.info("training on cpu")
    else:
        logging.info("training on gpu")

    logging.info("build vocab")
    src_voc = Vocabulary(strain_path, svoc_path, ["<pad>", "<eos>", "<unk>"])
    tgt_voc = Vocabulary(ttrain_path, tvoc_path, ["<pad>", "<sos>", "<eos>", "<unk>"])
    svocab_size = src_voc.voc_len()
    tvocab_size = tgt_voc.voc_len()

    logging.info("build model")
    if args.model == "cgru":
        model = cgru.Model(svocab_size, tvocab_size)
    model = model.cuda()

    # initial the parameters of model
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint))
    else:
        initial_params(model)

    # the structure of model and the number of model parameters
    logging.info(model)
    k = 0
    for i in list(model.parameters()):
        l = 1
        for j in i.size():
            l *= j
        k += l
    logging.info("total of parameters:" + str(k))

    # train
    # optimizer:adam, adadelta, sgd
    if args.model == "cgru":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_func = nn.NLLLoss(ignore_index=PAD)

    logging.info("train model")
    train(src_voc, tgt_voc, model, loss_func, optimizer)










