from enum import Flag
import torch
import torch.nn as nn
from flash_pytorch import FLASHTransformer
from g_mlp_pytorch import gMLP


def list2TensorWith_T_CAt(embedding_list, t):
    # embedding_list dim*len
    for i in range(len(embedding_list)):
        embedding_list[i] = embedding_list[i].T.unsqueeze(
            0) if t else embedding_list[i].unsqueeze(0)
    embedding_tensor = torch.cat(embedding_list, 0)
    return embedding_tensor


class myFlashTransformer(nn.Module):
    def __init__(self, num_tokens, dropout, pool_dim, kernel_size, stride, encoderConv, conv_dim, convPoolLen,
                 FLASHT_dim, FLASHT_depth, group_size, l1, l2, l3, l4):
        super().__init__()
        self.encoder = FLASHTransformer(
            num_tokens=num_tokens,
            dim=FLASHT_dim,
            depth=FLASHT_depth,
            causal=True,
            group_size=group_size)
        self.convEncoder = nn.Conv1d(in_channels=num_tokens,
                                     out_channels=encoderConv,
                                     stride=stride,
                                     kernel_size=kernel_size,
                                     padding='same')
        self.LN = nn.LayerNorm(encoderConv)
        self.pool = nn.AdaptiveMaxPool1d(pool_dim)
        self.conv = nn.Conv1d(in_channels=encoderConv,
                              out_channels=conv_dim,
                              stride=stride,
                              kernel_size=kernel_size,
                              padding='same')
        self.convPool = nn.AdaptiveMaxPool1d(convPoolLen)
        self.BN = nn.BatchNorm1d(conv_dim)
        self.dropout = nn.AlphaDropout(dropout)
        self.l1 = nn.Linear(conv_dim, l1)
        self.l2 = nn.Linear(convPoolLen, l2)
        self.l3 = nn.Linear(l2, l3)
        self.l4 = nn.Linear(l3, l4)
        self.sigmoidLayer = nn.Sigmoid()

    def forward(self, seq):
        relu = nn.RReLU()
        elu = nn.ELU()
        encoder = [self.encoder(i) for i in seq]  # 1*L*dim
        convEncoder_out = [
            elu(self.LN(self.convEncoder(i.transpose(1, 2)).transpose(1, 2))) for i in encoder]# 1*L*dim
        pool_out = [self.pool(i.squeeze(0).T)
                    for i in convEncoder_out]  # dim*pool_len
        embedding = list2TensorWith_T_CAt(pool_out, t=False)  # batch*dim*len
        conv_out = self.conv(embedding)  # batch*convOutDim*len
        convPool_out = self.convPool(conv_out)  # batch*convOutDim*convPoolLen
        BN_out = self.BN(relu(convPool_out)
                         )  # batch*convOutDim*len
        l1_out = self.l1(BN_out.transpose(1, 2)) # batch*len*dim
        l2_out = self.l2(l1_out.squeeze(2))
        l3_out = self.l3(l2_out)
        l4_out = self.l4(l3_out)
        output = self.sigmoidLayer(l4_out)
        return output


class gmlp(nn.Module):
    def __init__(self, num_tokens, dropout, pool_dim, kernel_size, stride, encoderConv, conv_dim, convPoolLen,
                 gmlp_dim, gmlp_depth, seq_len, l1, l2, l3, l4):
        super().__init__()
        self.encoder = gMLP(
            num_tokens=num_tokens,
            dim=gmlp_dim,
            depth=gmlp_depth,
            seq_len=seq_len,  # use circulant weight matrix for linear increase in parameters in respect to sequence length
            causal=True,
            circulant_matrix=True,
            heads=6)  # 4 heads
        self.convEncoder = nn.Conv1d(in_channels=num_tokens,
                                     out_channels=encoderConv,
                                     stride=stride,
                                     kernel_size=kernel_size,
                                     padding='same')
        self.LN_ = nn.LayerNorm(encoderConv)
        self.pool = nn.AdaptiveMaxPool1d(pool_dim)
        self.conv = nn.Conv1d(in_channels=encoderConv,
                              out_channels=conv_dim,
                              stride=stride,
                              kernel_size=kernel_size,
                              padding='same')
        self.convPool = nn.AdaptiveMaxPool1d(convPoolLen)
        self.BN_ = nn.BatchNorm1d(conv_dim)
        self.dropout = nn.AlphaDropout(dropout)
        self.l1 = nn.Linear(conv_dim, l1)
        self.l2 = nn.Linear(convPoolLen, l2)
        self.l3 = nn.Linear(l2, l3)
        self.l4 = nn.Linear(l3, l4)
        self.sigmoidLayer = nn.Sigmoid()

    def forward(self, seq):
        relu = nn.RReLU()
        elu = nn.ELU()
        encoder = [self.encoder(i) for i in seq]  # 1*L*dim
        convEncoder_out = [
            elu(self.LN_(self.convEncoder(i))) for i in encoder] # 1*L*dim
        pool_out = [self.pool(i.squeeze(0).T)
                    for i in convEncoder_out]  # dim*pool_len
        embedding = list2TensorWith_T_CAt(pool_out, t=False)  # batch*dim*len
        conv_out = self.conv(embedding)  # batch*convOutDim*len
        convPool_out = self.convPool(conv_out)  # batch*convOutDim*convPoolLen
        BN_out = self.BN_(relu(convPool_out).transpose(1, 2)
                         )  # batch*len*convOutDim
        l1_out = self.l1(BN_out)
        l2_out = self.l2(l1_out.squeeze(2))
        l3_out = self.l3(l2_out)
        l4_out = self.l4(l3_out)
        output = self.sigmoidLayer(l4_out)
        return output
