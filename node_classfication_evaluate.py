import numpy as np
import scipy.io as sio
import pickle as pkl
import torch.nn as nn
from sklearn.metrics import f1_score
import time
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.logreg import LogReg


def load_data(dataset, datasetfile_type):
    """"Get the label of node classification, training set, verification machine and test set"""
    if datasetfile_type == 'mat':
        data = sio.loadmat('data/{}.mat'.format(dataset))
    else:
        data = pkl.load(open('data/{}.pkl'.format(dataset), "rb"))
    try:
        labels = data['label']
    except:
        labels = data['labelmat']

    idx_train = data['train_idx'].ravel()
    try:
        idx_val = data['valid_idx'].ravel()
    except:
        idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    return labels, idx_train.astype(np.int32) - 1, idx_val.astype(np.int32) - 1, idx_test.astype(np.int32) - 1

# def visualize_t_sne(embeddings, node_labels):
#     # 转换为numpy数组并提取节点特征向量
#     embeddings_np = embeddings.squeeze().detach().cpu().numpy()
#     node_labels_np = node_labels.squeeze().detach().cpu().numpy()
#
#     # 使用 t-SNE 进行降维
#     tsne = TSNE(n_components=2, random_state=42)
#     embeddings_tsne = tsne.fit_transform(embeddings_np)
#     norm = plt.Normalize(node_labels_np.min(), node_labels_np.max())
#     cmap = plt.cm.get_cmap('viridis')
#     node_colors = cmap(norm(node_labels_np))
#
#     # 绘制 t-SNE 可视化图形
#     plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=node_labels)
#     plt.title("t-SNE Visualization of Node Classification Results")
#     plt.show()



def node_classification_evaluate(model, feature, A, file_name, file_type, device, isTest=True):
    """Node classification training process"""
    # features_fused = features_fused.to(device)
    # adj_fused = adj_fused.to(device)

    embeds = model(feature, A)

    labels, idx_train, idx_val, idx_test = load_data(file_name, file_type)

    try:
        labels = labels.todense()
    except:
        pass
    labels = labels.astype(np.int16)
    embeds = torch.FloatTensor(embeds[np.newaxis]).to(device)
    labels = torch.FloatTensor(labels[np.newaxis]).to(device)
    idx_train = torch.LongTensor(idx_train).to(device)
    idx_val = torch.LongTensor(idx_val).to(device)
    idx_test = torch.LongTensor(idx_test).to(device)


    hid_units = embeds.shape[2]
    nb_classes = labels.shape[2]
    xent = nn.CrossEntropyLoss()
    train_embs = embeds[0, idx_train]
    val_embs = embeds[0, idx_val]
    test_embs = embeds[0, idx_test]
    train_lbls = torch.argmax(labels[0, idx_train], dim=1)
    val_lbls = torch.argmax(labels[0, idx_val], dim=1)
    test_lbls = torch.argmax(labels[0, idx_test], dim=1)

    accs = []
    micro_f1s = []
    macro_f1s = []
    macro_f1s_val = []
    for _ in range(1):
        log = LogReg(hid_units, nb_classes)
        opt = torch.optim.Adam([{'params': model.parameters(), 'lr': 0.005}, {'params': log.parameters()}], lr=0.005, weight_decay=0.0005)
        log.to(device)

        val_accs = []
        test_accs = []
        val_micro_f1s = []
        test_micro_f1s = []
        val_macro_f1s = []
        test_macro_f1s = []

        starttime = time.time()
        for iter_ in range(100):
            embeds = model(feature, A)
            # print(embeds.shape)
            embeds = torch.FloatTensor(embeds[np.newaxis]).to(device)  # np.newaxis 的功能是增加新的维度，但是要注意 np.newaxis 放的位置不同，产生的矩阵形状也不同 通常用它将一维的数据转换成一个矩阵，这样就可以与其他矩阵进行相乘
            train_embs = embeds[0, idx_train]
            val_embs = embeds[0, idx_val]
            test_embs = embeds[0, idx_test]

            # train
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

            logits_val = log(val_embs)
            preds = torch.argmax(logits_val, dim=1)

            val_acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]
            val_f1_macro = f1_score(val_lbls.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='macro')   #返回一个新的tensor，从当前计算图中分离下来。但是仍指向原变量的存放位置，不同之处只是requirse_grad为false.得到的这个tensir永远不需要计算器梯度，不具有grad
            val_f1_micro = f1_score(val_lbls.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='micro')

            print("{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(iter_ + 1, loss.item(), val_acc, val_f1_macro,
                                                              val_f1_micro))
            # print("weight_b:{}".format(model.weight_b))
            # print("weight_a:{}".format(model.weight_a))
            # print("weight_c:{}".format(model.weight_c))

            val_accs.append(val_acc.item())
            val_macro_f1s.append(val_f1_macro)
            val_micro_f1s.append(val_f1_micro)

            # test
            logits_test = log(test_embs)
            preds = torch.argmax(logits_test, dim=1)

            test_acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            test_f1_macro = f1_score(test_lbls.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='macro')
            test_f1_micro = f1_score(test_lbls.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='micro')
            print("test_f1-ma: {:.4f}\ttest_f1-mi: {:.4f}".format(test_f1_macro, test_f1_micro))
            print('=' * 50)

            test_accs.append(test_acc.item())
            test_macro_f1s.append(test_f1_macro)
            test_micro_f1s.append(test_f1_micro)

        # visualize_t_sne(embeds,labels)
        # 在最后一次迭代之后插入 t-SNE 可视化代码
        # t-SNE visualization
        # tsne = TSNE(n_components=2, random_state=42)
        # embeddings_tsne = tsne.fit_transform(embeds.squeeze().detach().cpu().numpy())
        #
        # # Plot t-SNE visualization
        # plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=train_lbls.cpu().detach().numpy())
        # plt.title("t-SNE Visualization of Node Classification Results")
        # plt.show()

        endtime = time.time()

        print('time: {:.10f}'.format(endtime - starttime))

        max_iter = val_accs.index(max(val_accs))
        accs.append(test_accs[max_iter])

        max_iter = val_macro_f1s.index(max(val_macro_f1s))
        macro_f1s.append(test_macro_f1s[max_iter])

        max_iter = val_micro_f1s.index(max(val_micro_f1s))
        micro_f1s.append(test_micro_f1s[max_iter])

    if isTest:
        print("\t[Classification] Macro-F1: {:.4f} ({:.4f}) | Micro-F1: {:.4f} ({:.4f})".format(np.mean(macro_f1s),
                                                                                                np.std(macro_f1s),
                                                                                                np.mean(micro_f1s),
                                                                                                np.std(micro_f1s)))
    else:
        return np.mean(macro_f1s), np.mean(micro_f1s)

    return np.mean(macro_f1s), np.mean(micro_f1s)