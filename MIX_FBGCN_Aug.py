import torch
from torch import cat
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F

from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split
from Evaluator import LREvaluator
from GCL.models import DualBranchContrast
from utility.data import build_graph, adj_lap
from utility.config import get_arguments
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import get_laplacian


def adj_lap(edge_index, num_nodes, device):
    edge_index_adj, adj_weight= gcn_norm(edge_index, None, num_nodes, add_self_loops=False)
    edge_index_lap, lap_weight= get_laplacian(edge_index, None, "sym")
    shape = num_nodes
    adj_length = adj_weight.size()[0]
    lap_length = lap_weight.size()[0]
    adj = torch.zeros(shape, shape).to(device)
    lap = torch.zeros(shape, shape).to(device)

    for i in range(adj_length):
        x1 = edge_index_adj[0][i]
        y1 = edge_index_adj[1][i]
        adj[x1][y1] = adj_weight[i]
    for i in range(lap_length): 
        x2 = edge_index_lap[0][i]
        y2 = edge_index_lap[1][i]
        lap[x2][y2] = lap_weight[i]
    return lap, adj

class Pre_Mix_Layer(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.encoder = torch.nn.Linear(in_dim, out_dim, bias = False)
        self.reset_parameters()
    
    def reset_parameters(self):
        gain = torch.nn.init.calculate_gain("relu")
        torch.nn.init.xavier_normal_(self.encoder.weight, gain)

    def forward(self, x, Lsym, Anorm, encode="low"):
        if encode == "high":
            Lhp = Lsym
            Hh = F.relu(torch.mm(Lhp, self.encoder(x)))      
            return Hh
        else:
            Llp = Anorm
            Hl = F.relu(torch.mm(Llp, self.encoder(x)))      
            return Hl


class Pre_Train(torch.nn.Module):
    def __init__(self, n_layer, in_dim, hi_dim, out_dim, dropout):
        """
        :param n_layer: number of layers
        :param in_dim: input dimension
        :param hi_dim: hidden dimension
        :param out_dim: output dimension
        :param dropout: dropout ratio
        """
        super().__init__()
        assert(n_layer > 0)
        self.num_layers = n_layer
        self.stacks = torch.nn.ModuleList()
        # first layer
        self.stacks.append(Pre_Mix_Layer(in_dim, hi_dim))
        # inner layers
        # for _ in range(n_layer - 2):
        #     self.stacks.append(Pre_Mix_Layer(hi_dim, hi_dim))
        # last layer
        self.stacks.append(Pre_Mix_Layer(hi_dim, out_dim))
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for hplayer in self.stacks:
            hplayer.reset_parameters()

    def forward(self, x, lsym, anorm, encode="low"):
        if encode == "high":
            x = F.relu(self.stacks[0](x, lsym, anorm, encode))
            x = F.dropout(x, p=self.dropout, training=self.training)
            return self.stacks[-1](x, lsym, anorm, encode)
        else:
            x = F.relu(self.stacks[0](x, lsym, anorm, encode))
            x = F.dropout(x, p=self.dropout, training=self.training)
            return self.stacks[-1](x, lsym, anorm, encode)


class Encoder(torch.nn.Module):
    def __init__(self, pretrain_model, hidden_dim, proj_dim, device, augmentor=None):
        super(Encoder, self).__init__()
        self.pretrain_model = pretrain_model
        self.augmentor = augmentor
        self.device = device

        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, hidden_dim)

    def forward(self, data):
        aug1, aug2 = self.augmentor
        if aug1 != None:
            x1, edge_index1, edge_weight1 = aug1(data.x, data.edge_index, None)
            x2, edge_index2, edge_weight2 = aug2(data.x, data.edge_index, None)
        lsym1, anorm1 = adj_lap(edge_index1, data.num_nodes, self.device)
        lsym2, anorm2 = adj_lap(edge_index2, data.num_nodes, self.device)

        z1 = self.pretrain_model(x1, lsym1, anorm1, "high")
        z2 = self.pretrain_model(x2, lsym2, anorm2, "low")
        # z = cat((z1, z2), 1)
        z = z2
        return z, z1, z2

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

def train(encoder_model, contrast_model, data, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    z, z1, z2 = encoder_model(data)
    h1, h2 = [encoder_model.project(x) for x in [z1, z2]]
    loss = contrast_model(h1, h2)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(encoder_model, data):
    encoder_model.eval()
    z, _, _ = encoder_model(data)
    split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = LREvaluator()(z, data.y, split)
    return result


def main():
    args = get_arguments()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    hidden_dim = args.hidden_dim
    second_hidden_dim = 128
    device = torch.device('cuda')
    dataset = "cora"
    if(args.loss_type == "False"):
        loss_type = False
    else:
        loss_type = True
    data = build_graph(dataset).to(device)
    aug1 = A.Compose([A.EdgeRemoving(pe=0.3), A.FeatureMasking(pf=0.3)])
    aug2 = A.Compose([A.EdgeRemoving(pe=0.3), A.FeatureMasking(pf=0.3)])
    fbconv = Pre_Train(2, data.num_features, hidden_dim, second_hidden_dim, 0.5)
    encoder_model = Encoder(pretrain_model=fbconv, hidden_dim=128, proj_dim=128, device=device,augmentor=(aug1, aug2)).to(device)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs=loss_type ).to(device)

    optimizer = Adam(encoder_model.parameters(), lr=args.pre_learning_rate, weight_decay=5e-5)

    with tqdm(total=args.preepochs, desc='(T)') as pbar:
        for epoch in range(args.preepochs):
            loss = train(encoder_model, contrast_model, data, optimizer)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    test_result = test(encoder_model, data)
    print(f'(E): MIX_FBGCN: Best test accuracy={test_result["accuracy"]:.4f}')


if __name__ == '__main__':
    main()