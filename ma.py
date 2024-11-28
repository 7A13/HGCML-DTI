import psutil
from utilsdtiseed import *
from tqdm import tqdm
import torch
import torch.nn.functional as F
import warnings
from data import load_hetero_data
from model import HAN

warnings.filterwarnings("ignore")
seed = 47
args = setup(default_configure,seed)
s = 47
in_size = 256
hidden_size = 128
out_size = 128
dropout = 0.5
lr = 0.001
weight_decay = 1e-10
epochs = 1000
cl_loss_co = 1
reg_loss_co = 0.0001
fold = 0
dir = "../modelSave"
args['device'] = "cuda:0" if torch.cuda.is_available() else "cpu"

for name in ["heter"]:
# for name in ["heter","Es","GPCRs","ICs","Ns","zheng"]:
#     dtidata, graph, num, all_meta_paths = load_dataset(name)
    dateset, graph,edge ,cledg= load_hetero_data()
    # dataName heter Es GPCRs ICs Ns zheng
    dti_label = torch.tensor(dateset[:, 2:3]).to(args['device'])
    hd = torch.randn((708, in_size))
    hp = torch.randn((1512, in_size))

    # hd = torch.eye(1094)
    # hp = torch.eye(1556)

    features_d = hd.to(args['device'])
    features_p = hp.to(args['device'])
    node_feature = [features_d, features_p]

    dti_cl = get_clGraph(dateset, cledg).to(args['device'])

    cl = dti_cl
    data = dateset
    label=dti_label
    train_indeces, test_indeces = get_cross(dateset)



    def main(seed):

        all_acc = []
        all_roc = []
        all_f1 = []
        for i in range(len(train_indeces)):
            f = open(f"{i}foldtrain.txt","w",encoding="utf-8")
            train_index = train_indeces[i]
            for train_index_one in train_index:
                f.write(f"{train_index_one}\n")
            test_index = test_indeces[i]
            f = open(f"{i}foldtest.txt","w",encoding="utf-8")
            for train_index_one in test_index:
                f.write(f"{train_index_one}\n")
            #
            # if not os.path.isdir(f"{dir}"):
            #     os.makedirs(f"{dir}")
            model = HAN(num_meta_paths=[5,4],
                        in_size=[node_feature[0].shape[1],node_feature[1].shape[1]],
                        hidden_size=128,
                        out_size=128,
                        num_heads=1,
                        k_layers=2,
                        alpha=0.05,
                        edge_drop=0.005,
                        dropout=0.1
                        )
            model = model.to(args['device'])
            # model.load_state_dict(torch.load(f"{dir}/net{i}.pth"))
            optim = torch.optim.Adam(lr=lr, weight_decay=weight_decay, params=model.parameters())
            best_acc = 0
            best_f1 = 0
            best_roc = 0
            for epoch in tqdm(range(epochs)):
                loss, train_acc, task1_roc, acc, task1_roc1, task1_pr = train(model, optim,train_index,test_index, epoch,i)
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

        print(f"{name},{sum(all_acc) / len(all_acc):.4f},  {sum(all_roc) / len(all_roc):.4f} ,{sum(all_f1) / len(all_f1):.4f}")

    def train(model, optim,train_index,test_index, epoch,fold):
        # train_index=torch.Tensor(train_index)
        # test_index = torch.Tensor(test_index)
        model.train()
        out, d, p = model(graph, node_feature,data,train_index,edge)

        train_acc = (out.argmax(dim=1) == label[train_index].reshape(-1)).sum(dtype=float) / len(train_index)

        task1_roc = get_roc(out, label[train_index])

        reg = get_L2reg(model.parameters())

        loss = F.nll_loss(out, label[train_index].long().reshape(-1)) # + reg_loss_co * reg

        optim.zero_grad()
        loss.backward()
        optim.step()
        print(u'1当前进程的内存使用:%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
        # print(f"{epoch} epoch loss  {loss:.4f} train is acc  {train_acc:.4f}, task1 roc is {task1_roc:.4f},")
        te_acc, te_task1_roc1, te_task1_pr = main_test(model, d, p,test_index, epoch,fold)
        print(f"{epoch} epoch loss  {loss:.4f} ,test_task1 roc is {te_task1_roc1:.4f},te_task1_pr is {te_task1_pr:.4f}")
        return loss.item(), train_acc, task1_roc, te_acc, te_task1_roc1, te_task1_pr



    def main_test(model, d, p, test_index ,epoch,fold):
        model.eval()

        out, d, p = model(graph, node_feature,data, test_index ,edge)

        acc1 = (out.argmax(dim=1) == label[test_index].reshape(-1)).sum(dtype=float) / len(test_index)

        task_roc = get_roc(out, label[test_index])

        task_pr = get_pr(out,label[test_index])
        # if epoch == 999:
        #     f = open(f"{fold}out.txt","w",encoding="utf-8")
        #     for o in  (out.argmax(dim=1) == label[test_index].reshape(-1)):
        #         f.write(f"{o}\n")
        #     f.close()
        return acc1, task_roc, task_pr

    main(seed)
