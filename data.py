import numpy as np
import scipy
import pickle
import torch
import dgl
from utilsdtiseed import *
import torch.nn.functional as F


def load_ACM_data(prefix=r'C:\Users\Yanyeyu\Desktop\实验2\HPN\dataset/ACM'):

    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()#节点类型0的特征，4019行4000列

    labels = np.load(prefix + '/labels.npy')#加载标签，4019
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')#加载训练集，验证集，测试集的索引

    # PAP = scipy.sparse.load_npz(prefix + '/pap.npz')
    # PSP = scipy.sparse.load_npz(prefix + '/psp.npz')
    PAP = np.load(prefix + '/PAP_only_one.npy')
    PSP = np.load(prefix + '/PSP_only_one.npy')
    PAP = scipy.sparse.csr_matrix(PAP)
    PSP = scipy.sparse.csr_matrix(PSP)
    g1 = dgl.DGLGraph(PAP)
    g2 = dgl.DGLGraph(PSP)
    g=[g1, g2]
    features = torch.FloatTensor(features_0)
    labels=torch.LongTensor(labels)
    num_classes = 3
    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']

    return g, features, labels, num_classes, train_idx, val_idx, test_idx



def load_IMDB_data(prefix=r'F:\Desktop\duibi\新建文件夹\IMDB'):
    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()#节点类型0的特征，4019行4000列

    labels = np.load(prefix + '/labels.npy')#加载标签，4019
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')#加载训练集，验证集，测试集的索引

    MAM = np.load(prefix + '/mam.npy')
    MAM = scipy.sparse.csr_matrix(MAM)
    MDM = np.load(prefix + '/mdm.npy')
    MDM = scipy.sparse.csr_matrix(MDM)


    g1 = dgl.DGLGraph(MAM)
    g2 = dgl.DGLGraph(MDM)
    g=[g1,g2]
    features = torch.FloatTensor(features_0)
    labels=torch.LongTensor(labels)
    num_classes=3
    train_idx=np.load(prefix + '/train_idx_0.9.npy')
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']


    return g, features, labels, num_classes, train_idx, val_idx, test_idx


def load_DBLP_data(prefix=r'E:\图神经网络\图神经网络\模型及代码\实验2\HPN\dataset\DBLP'):
    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()#节点类型0的特征，4019行4000列

    labels = np.load(prefix + '/labels.npy')#加载标签，4019
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')#加载训练集，验证集，测试集的索引

    APA = scipy.sparse.load_npz(prefix + '/apa_only_one.npz').A
    APA = scipy.sparse.csr_matrix(APA)
    APCPA = scipy.sparse.load_npz(prefix + '/apcpa_only_one.npz').A
    APCPA = scipy.sparse.csr_matrix(APCPA)
    APTPA = scipy.sparse.load_npz(prefix + '/aptpa_only_one.npz').A
    APTPA = scipy.sparse.csr_matrix(APTPA)



    g1 = dgl.DGLGraph(APA)
    g2 = dgl.DGLGraph(APCPA)
    g3 = dgl.DGLGraph(APTPA)
    g=[g1,g2,g3]
    #g = [g1, g2]
    features = torch.FloatTensor(features_0)
    labels=torch.LongTensor(labels)
    num_classes=4
    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']


    return g, features, labels, num_classes, train_idx, val_idx, test_idx
def load_hetero_data(prefix=r'D:\Users\86135\Desktop\data\heter'):
    drug_drug = np.loadtxt(prefix + '\mat_drug_drug.txt')#(708, 708
    drug_chemical = np.loadtxt(prefix + '\Similarity_Matrix_Drugs.txt')#(708, 708)
    drug_disease = np.loadtxt(prefix + '\mat_drug_disease.txt')#(708, 5603)
    drug_sideeffect = np.loadtxt(prefix  + '\mat_drug_se.txt')#(708, 4192)
    drug_drug_protein = np.loadtxt(prefix  + '\mat_drug_protein.txt')#(708, 1512)

    protein_protein = np.loadtxt(prefix  + '\mat_protein_protein.txt')#(1512, 1512)
    protein_protein_drug = drug_drug_protein.T
    protein_sequence = np.loadtxt(prefix  + '\Similarity_Matrix_Proteins.txt')#(1512, 1512)
    protein_disease = np.loadtxt(prefix  + '\mat_protein_disease.txt')#(1512, 5603)


    d_d=drug_drug
    d_d = torch.FloatTensor(d_d)
    d_d = F.normalize(d_d, dim=1, p=2)

    d_c=np.dot(drug_chemical,drug_chemical.T)
    d_c = torch.FloatTensor(d_c)
    d_c = F.normalize(d_c, dim=1, p=2)

    d_di =np.dot(drug_disease,drug_disease.T)
    d_di = torch.FloatTensor(d_di)
    d_di = F.normalize(d_di, dim=1, p=2)

    d_d_p=np.dot(drug_drug_protein,drug_drug_protein.T)
    d_d_p = torch.FloatTensor(d_d_p)
    d_d_p = F.normalize(d_d_p, dim=1, p=2)

    d_se=np.dot(drug_sideeffect,drug_sideeffect.T)
    d_se = torch.FloatTensor(d_se)
    d_se = F.normalize(d_se, dim=1, p=2)


    p_p=protein_protein
    p_p = torch.FloatTensor(p_p)
    p_p = F.normalize(p_p, dim=1, p=2)

    p_s=np.dot(protein_sequence,protein_sequence.T)
    p_s = torch.FloatTensor(p_s)
    p_s = F.normalize(p_s, dim=1, p=2)

    p_di=np.dot(protein_disease,protein_disease.T)
    p_di = torch.FloatTensor(p_di)
    p_di = F.normalize(p_di, dim=1, p=2)

    p_d_d=np.dot(protein_protein_drug,protein_protein_drug.T)
    p_d_d= torch.FloatTensor(p_d_d)
    p_d_d = F.normalize(p_d_d, dim=1, p=2)





    g=[[d_d,d_c,d_di,d_d_p,d_se],[p_p,p_s,p_di,p_d_d]]


    dti_o = np.loadtxt(prefix + '\drug_protein_train.txt')
    dti_test = np.loadtxt(prefix + '\drug_protein_test.txt')
    train_positive_index = []
    test_positive_index = []
    whole_negative_index = []

    for i in range(np.shape(dti_o)[0]):
        for j in range(np.shape(dti_o)[1]):
            if int(dti_o[i][j]) == 1:
                train_positive_index.append([i, j])

            elif int(dti_test[i][j]) == 1:
                test_positive_index.append([i, j])
            else:
                whole_negative_index.append([i, j])

    negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),
                                             size=len(test_positive_index) + len(train_positive_index),
                                             replace=False)
    # f = open(f"{time.strftime('%m_%d_%H_%M_%S', time.localtime())}_negtive.txt", "w", encoding="utf-8")
    # for i in negative_sample_index:
    #     f.write(f"{i}\n")
    # f.close()
    data_set = np.zeros((len(negative_sample_index) + len(test_positive_index) + len(train_positive_index), 3),
                        dtype=int)

    count = 0
    train_index = []
    test_index = []
    for i in train_positive_index:
        data_set[count][0] = i[0]
        data_set[count][1] = i[1]
        data_set[count][2] = 1
        train_index.append(count)
        count += 1
    for i in test_positive_index:
        data_set[count][0] = i[0]
        data_set[count][1] = i[1]
        data_set[count][2] = 1
        test_index.append(count)
        count += 1

    f = open(r"D:\Users\86135\Desktop\data\heter\dti_cledge.txt", "w", encoding="utf-8")
    for i in range(count):
        for j in range(count):
            if data_set[i][0] == data_set[j][0] or data_set[i][1] == data_set[j][1]:
                f.write(f"{i}\t{j}\n")

    for i in range(len(negative_sample_index)):
        data_set[count][0] = whole_negative_index[negative_sample_index[i]][0]
        data_set[count][1] = whole_negative_index[negative_sample_index[i]][1]
        data_set[count][2] = 0
        if i < 4000:
            train_index.append(count)
        else:
            test_index.append(count)
        count += 1
    f = open(r"D:\Users\86135\Desktop\data\heter\dti_index.txt", "w", encoding="utf-8")
    for i in data_set:
        f.write(f"{i[0]}\t{i[1]}\t{i[2]}\n")

    dateset = data_set
    a = np.zeros((3846, 3846))
    for i in range(len(dateset)):
        if dateset[i][2] == 1:
            a[dateset[i][0]][dateset[i][1]] = 1
    b = np.sum(a, 1)
    d=0
    for j in range(708):
        if b[j] == 0:
            for e in range(len(dateset)):
                if dateset[e][0]==j:
                    d = d + 1
                    print(j)

    f = open(r"D:\Users\86135\Desktop\data\heter\dtiedge.txt", "w", encoding="utf-8")
    for i in range(dateset.shape[0]):
        for j in range(i, dateset.shape[0]):
            if dateset[i][0] == dateset[j][0] or dateset[i][1] == dateset[j][1]:
                f.write(f"{i}\t{j}\n")

    f.close()
    edge = np.loadtxt(r"D:\Users\86135\Desktop\data\heter\dtiedge.txt", dtype=int)

    cledg = np.loadtxt(r"D:\Users\86135\Desktop\data\heter\dti_cledge.txt", dtype=int)


    return dateset, g,edge,cledg