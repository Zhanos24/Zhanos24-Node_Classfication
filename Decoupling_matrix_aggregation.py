import numpy as np
import torch
from scipy.sparse import coo_matrix

def coototensor(A):
    """
    Convert a coo_matrix to a torch sparse tensor
    """

    values = A.data
    indices = np.vstack((A.row, A.col))
    i = torch.LongTensor(indices)  # indices 表示 各个数据在各行的下标  indptr 表示每行数据的个数
    v = torch.FloatTensor(values)
    shape = A.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

# def adj_matrix_weight_merge(A, adj_weight, attention_scores):
def adj_matrix_weight_merge(A, adj_weight):
    """
    Multiplex Relation Aggregation
    """

    N = A[0][0].shape[0]
    temp = coo_matrix((N, N))
    temp = coototensor(temp)

    # Alibaba
    # a = coototensor(A[0][0].tocoo())
    # b = coototensor(A[1][0].tocoo())
    # c = coototensor(A[2][0].tocoo())
    # d = coototensor(A[3][0].tocoo())
    # A_t = torch.stack([a, b, c, d], dim=2).to_dense()

    # DBLP
    # a = coototensor(A[0][0].tocoo())
    # b = coototensor(A[0][1].tocoo())
    # c = coototensor(A[0][2].tocoo())
    # A_t = torch.stack([a, b, c], dim=2).to_dense()

    # Aminer
    # a = coototensor(A[0][0].tocoo())
    # b = coototensor(A[0][1].tocoo())
    # c = coototensor(A[0][2].tocoo())
    # A_t = torch.stack([a, c], dim=2).to_dense()

    ## IMDB
    a = coototensor(A[0][0].tocoo())
    b = coototensor(A[0][2].tocoo())
    A_t = torch.stack([a, b], dim=2).to_dense()


    temp = torch.matmul(A_t, adj_weight)  # 矩阵相乘
    temp = torch.squeeze(temp, 2)
    return temp + temp.transpose(0, 1)

    # temp = torch.matmul(A_t, adj_weight)  # 矩阵相乘
    # attention_temp = torch.matmul(A_t, attention_scores)
    # final_result = torch.matmul(temp, attention_temp)
    # final_result = torch.squeeze(final_result, 2)
    # return final_result + final_result.transpose(0, 1)






