import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
# from src.Utils import load_our_data2

from src.Decoupling_matrix_aggregation import adj_matrix_weight_merge

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features,
                                                  out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        try:
            input = input.float()
        except:
            pass
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)

        #=================================================================================
        # output = F.relu(output)
        # output = F.leaky_relu(output)
        # output = torch.tanh_(output)

        if self.bias is not None:
            return output + self.bias
        else:
            return output
    #
    # def __repr__(self):
    #     return self.__class__.__name__ + ' (' \
    #            + str(self.in_features) + ' -> ' \
    #            + str(self.out_features) + ')'


class GCN(nn.Module):
    """
    A Two-layer GCN.
    """

    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj, use_relu=True):
        x = self.gc1(x, adj)
        if use_relu:
            #=========================================================================================
            x = F.relu(x)
            # x = F.leaky_relu(x)
            # x = torch.tanh_(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)

        return x


class JRL_GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(JRL_GCN, self).__init__()
        """
        # stacked Graph Convolution
        """
        self.gc1 = GraphConvolution(nfeat, out)
        self.gc2 = GraphConvolution(out, out)
        # self.gc3 = GraphConvolution(out, out)
        # self.gc4 = GraphConvolution(out, out)
        # self.gc5 = GraphConvolution(out, out)
        self.dropout = dropout

        """
        Set the trainable weight of adjacency matrix aggregation
        """
        # Alibaba
        # self.weight_b = torch.nn.Parameter(torch.FloatTensor(4, 1), requires_grad=True)
        # torch.nn.init.uniform_(self.weight_b, a=0, b=0.14)

        # DBLP
        # self.weight_b = torch.nn.Parameter(torch.FloatTensor(3, 1), requires_grad=True)
        # torch.nn.init.uniform_(self.weight_b, a=0, b=0.14)

        # torch.nn.init.constant_(self.weight_b, 1)

        # Aminer
        # self.weight_b = torch.nn.Parameter(torch.FloatTensor(2, 1), requires_grad=True)
        # torch.nn.init.uniform_(self.weight_b, a=0, b=1)

        # IMDB
        self.weight_b = torch.nn.Parameter(torch.FloatTensor(2, 1), requires_grad=True)
        torch.nn.init.uniform_(self.weight_b, a=0, b=0.14)

        self.weight_a = torch.nn.Parameter(torch.FloatTensor(1, 1), requires_grad=True)
        torch.nn.init.uniform_(self.weight_a, a=0, b=0.01)


        self.attention_layer = MultiHeadAttention(input_dim=out, num_heads=4)

    def forward(self, feature, A, use_relu=True):
        # attention_scores, _ = self.attention_layer(features_fused)
        # final_A = adj_matrix_weight_merge(A, self.weight_b, attention_scores)

        final_A = adj_matrix_weight_merge(A, self.weight_b)

        # final_B = adj_matrix_weight_merge_b(A, self.weight_a)
        # final_C = torch.stack([final_A, final_B], dim=2)
        # final_C = torch.squeeze(final_C, 2)
        # final_C = final_C + final_C.transpose(0, 1)
        try:
            feature = torch.tensor(feature.astype(float).toarray())
        except:
            try:
                feature = torch.from_numpy(feature.toarray())
            except:
                pass

        # Output of single-layer GCN
        U1 = self.gc1(feature, final_A)
        # ================================================================================================
        # U1 = F.relu(U1)
        # U1 = F.leaky_relu(U1)
        # U1 = torch.tanh_(U1)
        # Output of two-layer GCN
        U2 = self.gc2(U1, final_A)

        # ================================================================================================
        # U2 = F.relu(U2)
        # U2 = F.leaky_relu(U2)
        # U2 = torch.tanh_(U2)

        # U3 = self.gc3(U2, final_A)
        # U4 = self.gc4(U2, final_A)
        # U5 = self.gc5(U2, final_A)

        # Average pooling
        float_a = self.weight_a.float()
        # float_c = self.weight_c.float()
        # U_final = U1 + U2

        # return U1
        # return (U1 + U2)/2
        return (U1 + U2 * float_a)/2
        # return (U1 + U2 + U3) / 3
        # return (U1 + U2 + U3 + U4) / 4
        # return (U1 + U2 + U3 + U4 + U5) / 5


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads, device):
        super(MultiHeadAttention, self).__init__()
        self.device = device
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.linear_q = nn.Linear(input_dim, input_dim).to(device)
        self.linear_k = nn.Linear(input_dim, input_dim).to(device)
        self.linear_v = nn.Linear(input_dim, input_dim).to(device)
        self.fc_out = nn.Linear(input_dim, input_dim).to(device)

    def forward(self, x):
        N, seq_length, _ = x.size()
        queries = self.linear_q(x).view(N, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.linear_k(x).view(N, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.linear_v(x).view(N, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(queries, keys.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attention = F.softmax(scores, dim=-1)

        out = torch.matmul(attention, values).transpose(1, 2).contiguous().view(N, seq_length, -1)
        return self.fc_out(out), attention


