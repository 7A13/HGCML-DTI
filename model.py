import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv, APPNPConv
from utilsdtiseed import *
from util_funcs import cos_sim
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import GraphConv
from GCNLayer import *
import torch.nn.functional as F
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class GraphConvolution(nn.Module):  # 自己定义的GCN
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.FloatTensor(out_features, out_features))

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):  # 这里的权重和偏置归一化
        #print(self.weight)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj):
        support = torch.spmm(inputs, self.weight)# HW in GCN
        adj = adj.to('cuda:0')
        output = torch.spmm(adj, support) # AHW
        if self.bias is not None:
            return F.elu(output + self.bias)
        else:
            return F.elu(output)



class SemanticAttention(nn.Module):  # 语义级注意力
    def __init__(self, in_size, hidden_size):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(nn.Linear(in_size, hidden_size),nn.Tanh(),nn.Linear(hidden_size, 1, bias=False))
    def forward(self, z):
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)  # 这里可以打印语义级注意力分配
        b = beta.expand((z.shape[0],) + beta.shape)
        return (b * z).sum(1)




class GCN(nn.Module):
    def __init__(self, nfeat, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, 256)
        self.gc2 = GraphConvolution(256, 256)
        self.dropout = dropout

    def forward(self, x, adj):
        x =  F.normalize(x, dim=1, p=2)
        # x = x.to(device)
        # adj = adj.to(device)
        x1 = F.relu(self.gc1(x, adj), inplace=True)
        x1 = F.dropout(x1, self.dropout)
        x2 = self.gc2(x1, adj)
        res = x2
        return res


class CL_GCN(nn.Module):
    def __init__(self, nfeat, dropout,alpha = 0.8):
        super(CL_GCN, self).__init__()
        self.gcn1 = GCN(nfeat, dropout)
        self.gcn2 = GCN(nfeat, dropout)
        self.gcn3 = GCN(nfeat, dropout)
        self.tau = 0.5
        self.alpha = alpha

    def forward(self, x1, adj1, x2, adj2,x3, adj3): \

        z1 = self.gcn1(x1, adj1)
        z2 = self.gcn2(x2, adj2)
        z3 = self.gcn3(x3, adj3)

        return z1, z2,z3
class MLP(nn.Module):
    def __init__(self, nfeat):
        super(MLP, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(nfeat, 128, bias=False).apply(init),
            nn.ELU(),
            nn.Linear(128, 2, bias=False),
            nn.LogSoftmax(dim=1))
            # nn.Sigmoid())
    def forward(self, x):
        output = self.MLP(x)
        return output
def init(i):
    if isinstance(i, nn.Linear):
        torch.nn.init.xavier_uniform_(i.weight)


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(nn.Linear(in_size, hidden_size),nn.Tanh(),nn.Linear(hidden_size, 1, bias=False))
    def forward(self, z):
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)
        #print(beta)
        #beta=torch.tensor([[0.],[1.]]).to('cuda:0')
        beta = beta.expand((z.shape[0],) + beta.shape)
        return (beta * z).sum(1)

class HANLayer(nn.Module):

    def __init__(self, num_meta_paths, hidden_size, k_layers, alpha, edge_drop, dropout):
        super(HANLayer, self).__init__()
        self.appnp_layers = nn.ModuleList()
        self.gat_layers= nn.ModuleList()
        self.gcnlayers = nn.ModuleList()

        for i in range(num_meta_paths):
            #self.gat_layers.append(GATConv(128, 128, 1,dropout, dropout, activation=F.elu))
            # 两层 alpha=0.03只能跑92
            ##self.appnp_layers.append(APPNP(k_layers=k_layers, alpha=alpha, edge_drop=edge_drop, dropout=dropout))
            self.gcnlayers.append(GraphConvolution(128, 128))
        self.semantic_attention = SemanticAttention(in_size=hidden_size)
        self.num_meta_paths = num_meta_paths

    def forward(self, gs, h):
        semantic_embeddings = []
        for i, g in enumerate(gs):
            #g= sparse.coo_matrix(g)
            # g=dgl.DGLGraph(g)
            #semantic_embeddings.append(self.appnp_layers[i](h, g).flatten(1))
            #semantic_embeddings.append(self.gat_layers[i](g, h).flatten(1))
            semantic_embeddings.append(self.gcnlayers[i](h, g).flatten(1))

        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)
        return self.semantic_attention(semantic_embeddings)





class HAN(nn.Module):
    def __init__(self, num_meta_paths, in_size, hidden_size, out_size, num_heads, k_layers, alpha, edge_drop, dropout):
        super(HAN, self).__init__()
        # 投影层
        self.fc_trans1=nn.Linear(in_size[0], hidden_size, bias=False)
        self.fc_trans2 = nn.Linear(in_size[1], hidden_size, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.layers1 = nn.ModuleList()
        self.layers1.append(HANLayer(num_meta_paths[0], hidden_size, k_layers, alpha, edge_drop, dropout))
        for l in range(1, num_heads):
            self.layers1.append(HANLayer(num_meta_paths[0], hidden_size, k_layers, alpha, edge_drop, dropout))

        self.layers2 = nn.ModuleList()
        self.layers2.append(HANLayer(num_meta_paths[1], hidden_size, k_layers, alpha, edge_drop, dropout))
        for l in range(1, num_heads):
            self.layers2.append(HANLayer(num_meta_paths[1], hidden_size, k_layers, alpha, edge_drop, dropout))
        self.predict = nn.Linear(hidden_size, out_size)
        self.CL_GCN = CL_GCN(256, dropout)
        self.attention = SemanticAttention(256, 256)
        self.MLP = MLP(256*3)


    def forward(self, g, h,data,dateset_index,edge):
        # out, d, p = model(graph, node_feature,data,train_index,edge)

        h1=self.fc_trans1(h[0])
        #h = self.dropout(h)
        for gnn1 in self.layers1:
            h1 = gnn1(g[0], h1)

        # h1 = self.dropout(h1)
        # h1=self.predict(h1)
        h2=self.fc_trans2(h[1])
        #h = self.dropout(h)
        for gnn2 in self.layers2:
            h2 = gnn2(g[1], h2)
        #h2 = self.dropout(h2)
        # h2 = self.predict(h2)

        # print(h1)
        # print(h2)
        # feature = torch.cat((h1[data[:, :1]], h2[data[:, 1:2]]), dim=2)
        # feature = feature.squeeze(1)
        # print(feature)
        # print("edgwg")
        #
        # edge = load_graph(np.array(edge), data.shape[0])


        edge, feature = constructur_graph(data, h1, h2,edge)

        f_edge, f_feature = constructure_knngraph(data,h1, h2)
        s_edge=edge*f_edge



        # feature1, feature2= self.CL_GCN(feature, edge, f_feature, f_edge)
        # print(feature1.shape)
        # print(feature2.shape)
        # pred = self.MLP(torch.cat((feature1, feature2), dim=1)[dateset_index])
        # simedge = F.softmax(torch.matmul(feature, feature.T), dim=-1)
        # # simedge = cos_sim(feature, feature)
        # sim =simedge * edge
        # simedge = torch.where(simedge < 0.5, torch.zeros_like(simedge), simedge)
        # print(sim)
        # print(simedge)
        # print("bian")
        feature1, feature2, feature3 = self.CL_GCN(feature, edge,feature, f_edge,feature,s_edge)
        # print(feature1)
        # print(feature2)
        # print(feature3)
        #
        # emb = torch.stack([feature1, feature2, feature3], dim=1)
        # emb = self.attention(emb)
        #
        emb=torch.cat((feature1, feature2,feature3), dim=1)
        pred = self.MLP(emb[dateset_index])

        return pred, h1,h2


class APPNP(nn.Module):
    # 0.03 0.1 0.0
    # yelp
    def __init__(self, k_layers, alpha, edge_drop, dropout=0.6):
        super(APPNP, self).__init__()
        self.appnp = APPNPConv(k_layers, alpha, edge_drop)
        self.dropout = nn.Dropout(p=dropout)
        # self.dropout = dropout
        # pass

    def forward(self, features, g):
        h = self.dropout(features)
        # h = F.dropout(features, self.dropout, training=self.training)
        h = self.appnp(g, h)
        return h