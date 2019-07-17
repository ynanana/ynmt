# cgru.py
# author: cyn
# email: yunachen@stu.xmu.edu.cn
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.autograd import Variable

from utils.nns import *
from utils.config import *


class Model(nn.Module):
    def __init__(self, svocab_size, tvocab_size, use_attention=True,
                 bidirectional=True):
        super(Model, self).__init__()
        # encoder
        self.svocab_size = svocab_size
        self.src_emb = nn.Embedding(self.svocab_size, args.emb_size,
                                    padding_idx=0)
        self.src_rnn = nn.GRU(args.emb_size, args.hidden_size,
                              batch_first=True, bidirectional=bidirectional)
        self.src_emb_drop = nn.Dropout(args.dropout)
        self.src_rnn_drop = nn.Dropout(args.dropout)

        # attention
        if use_attention:
            if bidirectional:
                hidden_size = args.hidden_size * 2
            else:
                hidden_size = args.hidden_size
            key_size = hidden_size
            query_size = args.hidden_size
            self.key = nn.Linear(key_size, args.hidden_size)
            self.query = nn.Linear(query_size, args.hidden_size)
            self.v = nn.Linear(args.hidden_size, 1)

        # decoder
        self.tvocab_size = tvocab_size
        self.tgt_emb = nn.Embedding(self.tvocab_size, args.emb_size,
                                    padding_idx=0)
        self.h_map = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size), nn.Tanh())
        self.tgt_rnn1 = nn.GRUCell(args.emb_size, args.hidden_size)
        self.tgt_rnn2 = nn.GRUCell(hidden_size, args.hidden_size)
        self.tgt_emb_drop = nn.Dropout(args.dropout)
        self.tgt_rnn_drop = nn.Dropout(args.dropout)

        self.read_out = nn.Linear(
            args.emb_size + args.hidden_size + hidden_size, args.emb_size * 2)
        self.out_drop = nn.Dropout(args.dropout)
        self.logit = nn.Linear(args.emb_size, self.tvocab_size)

    def encoder(self, x, x_len):
        src_embed = self.src_emb(x)
        src_embed = self.src_emb_drop(src_embed)
        src_embed_p = pack(src_embed, x_len, batch_first=True)
        encoder_output_p, h = self.src_rnn(src_embed_p, None)
        encoder_output, _ = unpack(encoder_output_p, batch_first=True)
        encoder_output = self.src_rnn_drop(encoder_output)
        return encoder_output, h

    def attention(self, encoder_output):
        key = self.key(encoder_output)
        value = encoder_output
        return key, value

    def decoder(self, y, h, mask, key, value):
        tgt_embed = self.tgt_emb(y)
        tgt_embed = self.tgt_emb_drop(tgt_embed)
        # decoder_output, _ = self.tgt_rnn(tgt_embed, h)

        max_len = y.shape[1]
        h_t = h

        decoder_output = []
        values = []
        for t in range(max_len):
            input = tgt_embed[:, t, :]  # b * s
            cell1 = self.tgt_rnn1(input, h_t)
            query = self.query(cell1.unsqueeze(1))
            a = self.v(torch.tanh(query + key)).squeeze(-1)  # b * t
            a.data.masked_fill_(mask, -float("inf"))
            a = F.softmax(a, -1)
            context = torch.bmm(a.unsqueeze(1), value).squeeze(1)
            h_t = self.tgt_rnn2(context, cell1)
            decoder_output.append(h_t)
            values.append(context)
        decoder_output = torch.stack(decoder_output, 1)
        values = torch.stack(values, 1)
        decoder_output = self.tgt_rnn_drop(decoder_output)

        return decoder_output, h_t, values, tgt_embed

    def output_layer(self, decoder_output, values, y):
        output = self.read_out(torch.cat([y, decoder_output, values], -1))
        output = maxout(output)
        output = self.out_drop(output)
        output = self.logit(output)
        output = F.log_softmax(output, dim=-1)

        output = output.view(decoder_output.shape[0],
                             decoder_output.shape[1], -1)

        return output

    def forward(self, x, y, x_len, y_len):
        y = F.pad(y, (1, 0, 0, 0), value=SOS)
        mask = (y != EOS).long()
        y = y * mask
        y = y[:, :-1]
        mask = x.eq(PAD)

        encoder_output, h = self.encoder(x, x_len)
        h = self.h_map(h[1])
        key, value = self.attention(encoder_output)
        decoder_output, _, values, inputs = self.decoder(y, h, mask, key, value)
        output = self.output_layer(decoder_output, values, inputs)

        return output

    def evaluate_(self, x, x_len):
        # Greedy search
        mask = x.eq(PAD)
        encoder_output, h = self.encoder(x, x_len)
        key, value = self.attention(encoder_output)
        h = self.h_map(h[1])

        max_len = x.shape[1] * 6
        input = SOS
        out = []
        for i in range(max_len):
            decoder_output, h, values, inputs = self.decoder(
                torch.LongTensor([input]).unsqueeze(0).cuda(),
                h, mask, key, value)
            output = self.output_layer(decoder_output, values, inputs)
            ps, indices = torch.topk(output, 1)
            out.append(indices.item())
            input = indices.item()
            if indices.item() == EOS:
                break
        print out
        return out

    def evaluate(self, x, x_len, beam_size=6):
        # beam search
        mask = x.eq(PAD)
        encoder_output, h = self.encoder(x, x_len)
        key, value = self.attention(encoder_output)

        sequence_ = [[SOS]]
        ps_ = [[0.0]]
        state_ = self.h_map(h[1])
        K_ = key
        V_ = value
        done = []

        def indices_mod(index):
            loop = index / self.tvocab_size
            index = index % self.tvocab_size
            return index, loop

        max_len = x.shape[1] * 6
        for i in range(max_len):
            input = torch.tensor(
                [seq[-1] for seq in sequence_]).unsqueeze(-1).cuda()
            decoder_output, h, valuess, inputs = self.decoder(
                input, state_, mask, K_, V_)
            output = self.output_layer(decoder_output, valuess, inputs)
            top_values, top_indices = torch.topk((output.squeeze(1) +
                                torch.tensor(ps_).cuda()).view(-1), beam_size)

            sequence = []
            state = []
            ps = []
            K = []
            V = []
            for (values, indices) in zip(top_values, top_indices):
                indices, beam = indices_mod(indices)
                if indices.item() == EOS or i == max_len - 1:
                    beam_size -= 1

                    done.append(sequence_[beam] + [indices.item()])
                else:
                    sequence.append(sequence_[beam] + [indices.item()])

                    ps.append([values.item()])
                    state.append(h[beam, :])
                    K.append(key[0])
                    V.append(value[0])
            if not sequence:
                break
            else:
                state = torch.stack(state, 0)
                K = torch.stack(K, 0)
                V = torch.stack(V, 0)

                sequence_ = sequence
                ps_ = ps
                state_ = state
                K_ = K
                V_ = V
        return done[0][1:]












