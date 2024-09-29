import numpy as np
import torch
from scipy.io import loadmat
from scipy.sparse import csr_matrix

from src.Model import JRL_GCN
from src.Model import MultiHeadAttention


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(
            np.int64))  # torch.from_numpy()方法把数组转换成张量，且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变。 np.vstack():在竖直方向上堆叠
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_our_data(dataset_str, cuda=True):
    """
    Load our Networks Datasets
    Avoid awful code
    """
    data = loadmat('data/' + dataset_str + '.mat')
    # label
    try:
        labels = data['label']
    except:
        labels = data['labelmat']
    N = labels.shape[0]  # 4685
    try:
        labels = labels.todense()
    except:
        pass

    # idx train valid test
    idx_train = data['train_idx'].ravel()
    try:
        idx_val = data['valid_idx'].ravel()
    except:
        idx_val = data['val_idx'].ravel()
    # idx_test = data['train_idx'].ravel()
    idx_test = data['test_idx'].ravel()
    # idx_train = np.concatenate((idx_train, idx_test))
    # node features

    try:
        node_features = data['full_feature'].toarray()
    except:
        try:
            node_features = data['feature']
        except:
            try:
                node_features = data['node_feature']
            except:
                node_features = data['features']
    features = csr_matrix(node_features)

    # edges to adj
    if dataset_str == 'small_alibaba_1_10':
        num_nodes = data['IUI_buy'].shape[0]
        adj = csr_matrix((num_nodes, num_nodes))
        adj = data['IUI_buy'] + data['IUI_cart'] + data["IUI_clk"] + data['IUI_collect']
    elif dataset_str == 'Aminer_10k_4class':
        num_nodes = 10000
        adj = csr_matrix((num_nodes, num_nodes))
        adj = data['PAP'] + data['PCP'] + data["PTP"]

        idx_test = idx_test - 1
        idx_train = idx_train - 1
        idx_val = idx_val - 1
    elif dataset_str == 'imdb_1_10':
        edges = data['edges'][0].tolist()
        num_nodes = edges[0].shape[0]
        adj = csr_matrix((num_nodes, num_nodes))
        for edge in edges:
            adj += edge
    else:
        num_nodes = data['A'][0][0].toarray().shape[0]
        adj = data['A'][0][0] + data['A'][0][1] + data['A'][0][2]

    print('{} node number: {}'.format(dataset_str, num_nodes))

    try:
        features = features.astype(np.int16)
    except:
        pass
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    idx_train = torch.LongTensor(idx_train.astype(np.int16))
    idx_val = torch.LongTensor(idx_val.astype(np.int16))
    idx_test = torch.LongTensor(idx_test.astype(np.int16))

    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()




    # 标签融合处理
    adj_fused, features_fused = fuse_labels_into_subgraphs(adj, labels, features, idx_train)

    if cuda:
        features_fused = features_fused.cuda()
        adj_fused = adj_fused.cuda()


    return adj_fused, features_fused, labels, idx_train, idx_val, idx_test
    
    adj = adj_fused
    features = features_fused
    # return adj, features, labels, idx_train, idx_val, idx_test

def load_our_data2(dataset, cuda):
    # 这里实现加载数据的逻辑
    adj, features, labels, idx_train, idx_val, idx_test = load_our_data(dataset, cuda)

    # 融合标签处理
    adj_fused, features_fused = fuse_labels_into_subgraphs(adj, labels, features, idx_train)
    if cuda:
        features_fused = features_fused.cuda()
        adj_fused = adj_fused.cuda()

    return adj_fused, features_fused, labels, idx_train, idx_val, idx_test



def get_model(model_opt, nfeat, nclass, A, nhid, out, dropout=0, cuda=True):
    """
     Model selection
    """
    if model_opt == "JRL_GCN":
        model = JRL_GCN(nfeat=nfeat,
                      nhid=nhid,
                      out=out,
                      dropout=dropout)
    elif model_opt == "SelfAttention":
        model = SelfAttention(nfeat=nfeat,
                      nhid=nhid,
                      out=out,
                      dropout=dropout)
    else:
        raise NotImplementedError('model:{} is not implemented!'.format(model_opt))

    if cuda: model.cuda()
    return model


def fuse_labels_into_subgraphs(adj, labels, features, idx_train):
    """
    Inject tags as a new node type into the subgraph and create corresponding edges.

    参数:
    - adj
    - labels
    - features
    - idx_train
    返回:
    - adj_fused
    - features_fused
    """
    # device = adj.device  # 获取 adj 的设备
    # features = features.to(device)  # 确保 features 在同一设备上
    # labels = labels.to(device)  # 确保 labels 在同一设备上
    
    num_nodes = adj.shape[0]  # 原始节点数量
    num_classes = labels.max().item() + 1


    label_adj = torch.zeros((num_nodes + num_classes, num_nodes + num_classes), device=adj.device)

    for node_idx in idx_train:
        label = labels[node_idx].item()
        label_adj[node_idx, num_nodes + label] = 1
        label_adj[num_nodes + label, node_idx] = 1


    adj_fused = torch.zeros((num_nodes + num_classes, num_nodes + num_classes), device=adj.device)
    adj_fused[:num_nodes, :num_nodes] = adj.to_dense()  # 将稀疏张量转为稠密张量
    adj_fused[num_nodes:, :num_nodes] = label_adj[num_nodes:, :num_nodes]
    adj_fused[:num_nodes, num_nodes:] = label_adj[:num_nodes, num_nodes:]


    features_fused = torch.cat([features, torch.ones((num_classes, features.shape[1]), device=features.device)], dim=0)  # 加入标签特征
    
    # # 打印设备信息，确保所有张量在同一设备上
    # # print(f"features_fused device: {features_fused.device}")
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # multihead_att = MultiHeadAttention(input_dim=features_fused.shape[1], num_heads=4, device=features_fused.device)
    # 
    # # 再次确保在调用前检查设备
    # # print(f"multihead_att weights device: {multihead_att.linear_q.weight.device}")
    # 
    # attention_output, attention_scores = multihead_att(features_fused.unsqueeze(0))
    # 
    # adj_weighted = adj_fused + attention_scores.squeeze(0)
    # 
    # return adj_weighted, features_fused, labels, idx_train


    return adj_fused, features_fused





