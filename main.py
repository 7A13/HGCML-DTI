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
    dti_label = torch.tensor(dateset[:, 2:3]).to(args['device'])
    hd = torch.randn((graph[0][0].shape[0], 256))
    hp = torch.randn((graph[1][0].shape[0],256))
    features_d = hd.to(args['device'])
    features_p = hp.to(args['device'])
    node_feature = [features_d, features_p]
    dti_cl = get_clGraph(dateset, "dti").to(args['device'])
    cl = dti_cl
    data = dateset
    label = dti_label
    print(data,label)
    wad

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
        #     os.makedirs(f"{dir}")

        model = HMTCL(
            all_meta_paths=all_meta_paths,
            in_size=[hd.shape[1], hp.shape[1]],
            hidden_size=[hidden_size, hidden_size],
            out_size=[out_size, out_size],
            dropout=dropout,
        ).to(args['device'])
        # model.load_state_dict(torch.load(f"{dir}/net{i}.pth"))
        optim = torch.optim.Adam(lr=lr, weight_decay=weight_decay, params=model.parameters())
        best_acc = 0
        best_f1 = 0
        best_roc = 0
        for epoch in tqdm(range(epochs)):
            loss, train_acc, task1_roc, acc, task1_roc1, task1_pr = train(model, optim, train_index, test_index, epoch,
                                                                          i)
            if acc > best_acc:
                best_acc = acc
            if task1_pr > best_f1:
                best_f1 = task1_pr
            if task1_roc1 > best_roc:
                best_roc = task1_roc1
                # torch.save(obj=model.state_dict(), f=f"{dir}/net.pth")
        all_acc.append(best_acc)
        all_roc.append(best_roc)
        all_f1.append(best_f1)
        print(f"fold{i}  auroc is {best_roc:.4f} aupr is {best_f1:.4f} ")

    print(
        f"{name},{sum(all_acc) / len(all_acc):.4f},  {sum(all_roc) / len(all_roc):.4f} ,{sum(all_f1) / len(all_f1):.4f}")


def train(model, optim, train_index, test_index, epoch, fold):
    model.train()
    out, cl_loss, d, p = model(graph, node_feature, cl, train_index, data)

    train_acc = (out.argmax(dim=1) == label[train_index].reshape(-1)).sum(dtype=float) / len(train_index)

    task1_roc = get_roc(out, label[train_index])

    reg = get_L2reg(model.parameters())

    loss = F.nll_loss(out, label[train_index].reshape(-1)) + cl_loss_co * cl_loss + reg_loss_co * reg

    optim.zero_grad()
    loss.backward()
    optim.step()
    # print(f"{epoch} epoch loss  {loss:.4f} train is acc  {train_acc:.4f}, task1 roc is {task1_roc:.4f},")
    te_acc, te_task1_roc1, te_task1_pr = main_test(model, d, p, test_index, epoch, fold)

    return loss.item(), train_acc, task1_roc, te_acc, te_task1_roc1, te_task1_pr


def main_test(model, d, p, test_index, epoch, fold):
    model.eval()

    out = model(graph, node_feature, cl, test_index, data, iftrain=False, d=d, p=p)

    acc1 = (out.argmax(dim=1) == label[test_index].reshape(-1)).sum(dtype=float) / len(test_index)

    task_roc = get_roc(out, label[test_index])

    task_pr = get_pr(out, label[test_index])
    # if epoch == 999:
    #     f = open(f"{fold}out.txt","w",encoding="utf-8")
    #     for o in  (out.argmax(dim=1) == label[test_index].reshape(-1)):
    #         f.write(f"{o}\n")
    #     f.close()
    return acc1, task_roc, task_pr


train_indeces, test_indeces = get_cross(dtidata)
main(train_indeces, test_indeces, seed)