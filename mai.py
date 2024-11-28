import argparse
import torch
from model import HAN
from tools import evaluate_results_nc
from pytorchtools import EarlyStopping
from data import load_ACM_data,load_IMDB_data,load_DBLP_data,load_zeng_data
import numpy as np
import random
from sklearn.metrics import f1_score
from utilsdtiseed import *

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()
    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')
    return accuracy, micro_f1, macro_f1

def main(args):
    #g, features, labels, num_classes, train_idx, val_idx, test_idx = load_ACM_data()
    dateset, graph = load_zeng_data()
    #g, features, labels, num_classes, train_idx, val_idx, test_idx = load_DBLP_data()

    dti_label = torch.tensor(dateset[:, 2:3]).to(args['device'])

    hd = torch.randn((graph[0][0].shape[0], 256))
    print(hd.shape)

    hp = torch.randn((graph[1][0].shape[0],256))
    features_d = hd.to(args['device'])
    features_p = hp.to(args['device'])

    node_feature = [features_d, features_p]

    dti_cl = get_clGraph(dateset, "dti").to(args['device'])

    cl = dti_cl
    data =dateset
    label = dti_label


    svm_macro_avg = np.zeros((7,), dtype=np.float)
    svm_micro_avg = np.zeros((7,), dtype=np.float)
    nmi_avg = 0
    ari_avg = 0
    print('start train with repeat = {}\n'.format(args['repeat']))
    for cur_repeat in range(args['repeat']):
        print('cur_repeat = {}   ==============================================================='.format(args['repeat']))
        all_acc = []
        all_roc = []
        all_f1 = []
        for i in range(len(tr)):
            f = open(f"{i}foldtrain.txt", "w", encoding="utf-8")
            train_index = tr[i]
            for train_index_one in train_index:
                f.write(f"{train_index_one}\n")
            test_index = te[i]
            f = open(f"{i}foldtest.txt", "w", encoding="utf-8")
            for train_index_one in test_index:
                f.write(f"{train_index_one}\n")
            #
            # if not os.path.isdir(f"{dir}"):
            #     os.makedirs(f"{dir}"),fmgnfjg
        model = HAN(num_meta_paths=len(g),
                    in_size=features.shape[1],
                    hidden_size=args['hidden_units'],
                    out_size=num_classes,
                    num_heads=args['num_heads'],
                    k_layers=args['k_layers'],
                    alpha=args['alpha'],
                    edge_drop=args['edge_drop'],
                    dropout=args['dropout'])
        model = model.to(args['device'])
        g = [graph.to(args['device']) for graph in g]

        early_stopping = EarlyStopping(patience=args['patience'], verbose=True,save_path='checkpoint/checkpoint_{}.pt'.format(args['dataset']))  # 提早停止，设置的耐心值为5
        loss_fcn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],weight_decay=args['weight_decay'])

        for epoch in range(args['num_epochs']):
            model.train()
            logits,h = model(graph, features)
            loss = loss_fcn(logits[train_idx], labels[train_idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            logits,h = model(g, features)
            val_loss = loss_fcn(logits[val_idx], labels[val_idx])
            test_loss = loss_fcn(logits[test_idx], labels[test_idx])
            print('Epoch{:d}| Train Loss:{:.4f}| Val Loss:{:.4f}| Test Loss:{:.4f}'.format(epoch + 1, loss.item(),val_loss.item(),test_loss.item()))
            early_stopping(val_loss.data.item(), model)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

        print('\ntesting...')
        model.load_state_dict(torch.load('checkpoint/checkpoint_{}.pt'.format(args['dataset'])))
        model.eval()
        logits,h = model(g, features)
        svm_macro, svm_micro, nmi, ari = evaluate_results_nc(h[test_idx].detach().cpu().numpy(), labels[test_idx].cpu().numpy(),int(labels.max()) + 1)  # 使用SVM评估节点
        svm_macro_avg = svm_macro_avg + svm_macro
        svm_micro_avg = svm_micro_avg + svm_micro
        nmi_avg += nmi
        ari_avg += ari
    svm_macro_avg = svm_macro_avg / args['repeat']
    svm_micro_avg = svm_micro_avg / args['repeat']
    nmi_avg /= args['repeat']
    ari_avg /= args['repeat']
    print('---\nThe average of {} results:'.format(args['repeat']))
    print('Macro-F1: ' + ', '.join(['{:.6f}'.format(macro_f1) for macro_f1 in svm_macro_avg]))
    print('Micro-F1: ' + ', '.join(['{:.6f}'.format(micro_f1) for micro_f1 in svm_micro_avg]))
    print('NMI: {:.6f}'.format(nmi_avg))
    print('ARI: {:.6f}'.format(ari_avg))
    print('all finished')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='这是我们基于GAT所构建的HAN模型')
    parser.add_argument('--dataset', default='ACM', help='数据集')
    parser.add_argument('--lr', default=0.004, help='学习率')
    parser.add_argument('--num_heads', default=[8], help='多头注意力数及网络层数')
    parser.add_argument('--hidden_units', default=64, help='隐藏层数（实际隐藏层数：隐藏层数*注意力头数）')
    parser.add_argument('--dropout', default=0.5, help='丢弃率')
    # ACM k=2 alpha=0.3 edge_drop=0.1时效果好
    parser.add_argument('--k_layers', default=1, help='appnp层数')
    parser.add_argument('--edge_drop', default=0.2, help='appnp丢边率')
    parser.add_argument('--alpha', default=0.05, help='appnp的残差连接的系数')
    parser.add_argument('--num_epochs', default=1000, help='最大迭代次数')
    parser.add_argument('--weight_decay', default=0, help='权重衰减')
    parser.add_argument('--patience', type=int, default=5, help='耐心值')
    parser.add_argument('--seed', type=int, default=123,help='随机种子')
    parser.add_argument('--device', type=str, default='cpu', help='使用cuda:0或者cpu')
    parser.add_argument('--repeat', type=int, default=1, help='重复训练和测试次数')
    args = parser.parse_args().__dict__
    set_random_seed(args['seed'])
    print(args)
    main(args)

