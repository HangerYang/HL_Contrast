import torch
import os.path as osp
import GCL.losses as L
import torch_geometric.transforms as T
from utility.config import get_arguments
from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split
from GCL.models import SingleBranchContrast
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import uniform
from utility.data import build_graph
from Evaluator import LREvaluator
import numpy as np


class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(GCNConv(input_dim, hidden_dim))
            else:
                self.layers.append(GCNConv(hidden_dim, hidden_dim))
            self.activations.append(nn.PReLU(hidden_dim))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for conv, act in zip(self.layers, self.activations):
            z = conv(z, edge_index, edge_weight)
            z = act(z)
        return z


class Encoder(torch.nn.Module):
    def __init__(self, encoder, hidden_dim):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.project = torch.nn.Linear(hidden_dim, hidden_dim)
        uniform(hidden_dim, self.project.weight)

    @staticmethod
    def corruption(x, edge_index):
        return x[torch.randperm(x.size(0))], edge_index

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        g = self.project(torch.sigmoid(z.mean(dim=0, keepdim=True)))
        zn = self.encoder(*self.corruption(x, edge_index))
        return z, g, zn


def train(encoder_model, contrast_model, data, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    z, g, zn = encoder_model(data.x, data.edge_index)
    loss = contrast_model(h=z, g=g, hn=zn)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(encoder_model, data):
    encoder_model.eval()
    z, _, _ = encoder_model(data.x, data.edge_index)
    split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = LREvaluator()(z, data.y, split)
    return result


def main():
    args = get_arguments()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda")
    dataset = args.dataset
    hidden_dim = args.hidden_dim
    pre_learning_rate = args.pre_learning_rate
    total_result = []
    
    for i in range(10):
        data = build_graph(dataset).to(device)
        gconv = GConv(input_dim=data.num_features, hidden_dim=hidden_dim, num_layers=2).to(device)
        encoder_model = Encoder(encoder=gconv, hidden_dim=hidden_dim).to(device)
        contrast_model = SingleBranchContrast(loss=L.JSD(), mode='G2L').to(device)

        optimizer = Adam(encoder_model.parameters(), lr=pre_learning_rate)

        with tqdm(total=args.preepochs, desc='(T)') as pbar:
            for epoch in range(args.preepochs):
                loss = train(encoder_model, contrast_model, data, optimizer)
                pbar.set_postfix({'loss': loss})
                pbar.update()

        test_result = test(encoder_model, data)
        total_result.append(test_result["accuracy"])

    with open('./results/nc_DGI_{}.csv'.format(args.dataset), 'a+') as file:
        file.write('\n')
        file.write('pretrain epochs = {}\n'.format(args.preepochs))
        file.write('pre_learning_rate = {}\n'.format(args.pre_learning_rate))
        file.write('hidden_dim = {}\n'.format(args.hidden_dim))
        file.write('(E): DGI Mean Accuracy: {}, with Std: {}'.format(np.mean(total_result), np.std(total_result)))


if __name__ == '__main__':
    main()