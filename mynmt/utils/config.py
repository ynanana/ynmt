# config.py
# author: cyn
# email: yunachen@stu.xmu.edu.cn
# coding=utf-8
import argparse
import os
import logging

PAD = 0
SOS = 1
EOS = 2
UNK = 3

# path
parser = argparse.ArgumentParser(description="NMT of cyn")
parser.add_argument("--train_data", type=str, default="", nargs=2,
                    help="path of train data")
parser.add_argument("--dev_data", type=str, default="", nargs=2,
                    help="path of dev data")
parser.add_argument("--test_data", type=str, default="", nargs=2,
                    help="path of test data")
parser.add_argument("--vocabulary", type=str, default="", nargs=2,
                    help="path of vocabulary")
parser.add_argument("--model", type=str, default="cgru",
                    help="name of model")

parser.add_argument("--train", type=bool, default=True,
                    help="trainig the model")

# embedding
parser.add_argument("--max_vocab_size", type=int, default=30000,
                    help="maximal vocabulary size")
parser.add_argument("--max_sequen_size", type=int, default=60,
                    help="maximal sequence size")

# network parameters
parser.add_argument("--emb_size", type=int, default=500,
                    help="embedding size for encoder and decoder input words/tokens")
parser.add_argument("--hidden_size", type=int, default=1000,
                    help="hidden size")
parser.add_argument("--model_dim", type=int, default=512,
                    help="model size")

# training parameters
parser.add_argument("--batch_size", type=int, default=80, help="batch size")
parser.add_argument("--learning_rate", type= float, default=0.001,
                    help="learning rate")

parser.add_argument("--lr_decay", type=float, default=0.05,
                    help="learning rate")
parser.add_argument("--grad_clip", type=float, default=1.0,
                    help="maximal gradient norm")
parser.add_argument("--dropout", type=float, default=0.1,
                    help="drop out")

parser.add_argument("--checkpoint", type=str, default="model.pkl",
                    help="name of checkpoint")

args = parser.parse_args()
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S', filemode='w')

