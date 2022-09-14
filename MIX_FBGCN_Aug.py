import torch
import GCL.losses as L
import torch.nn.functional as F

from torch_geometric.utils import get_laplacian
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split
from Evaluator import LREvaluator
from GCL.models import DualBranchContrast
from model.trial_pretrain import GCNConv
from utility.data import build_graph
from utility.config import get_arguments
from model.pretrain import get_augmentor



class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, second_hidden_dim, activation, num_layers):
        super(GConv, self).__init__()
        self.activation = activation()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim, cached=False))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, second_hidden_dim, cached=False))

    def forward(self, x, edge_index, edge_weight=None, normalize=True):
        z = x
        for i, conv in enumerate(self.layers):
            z = conv(z, edge_index, edge_weight, normalize)
            z = self.activation(z)
        return z

class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, proj_dim):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, hidden_dim)

    def forward(self, x, edge_index, edge_weight):
        aug1, aug2 = self.augmentor
        edge_index_high, edge_weight_high = get_laplacian(edge_index, edge_weight, normalization="sym")
        x1, edge_index1, edge_weight1 = aug1(x, edge_index_high, edge_weight_high)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        z = self.encoder(x, edge_index, edge_weight)
        z1 = self.encoder(x1, edge_index1, edge_weight1, normalize=False)
        z2 = self.encoder(x2, edge_index2, edge_weight2)
        return z, z1, z2

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)


def train(encoder_model, contrast_model, data, optimizer):
    encoder_model.train()
    optimizer.zero_grad()

    z, z1, z2 = encoder_model(data.x, data.edge_index, data.edge_weight)
    h1, h2 = [encoder_model.project(x) for x in [z1, z2]]
    loss = contrast_model(h1, h2)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(encoder_model, data):
    encoder_model.eval()
    z, _, _ = encoder_model(data.x, data.edge_index, data.edge_weight)
    split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = LREvaluator()(z, data.y, split)
    return result


def main():
    args = get_arguments()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda')
    dataset = args.dataset
    second_hidden_dim = 128
    if(args.loss_type == "False"):
        loss_type = False
    else:
        loss_type = True
    data = build_graph(dataset).to(device)
    one_side = (args.aug_side != "both")
    aug1, aug2 = get_augmentor(args.aug_type, one_side, args.aug_side, args.aug)
    gconv = GConv(data.num_features, args.hidden_dim, second_hidden_dim, activation=torch.nn.ReLU, num_layers=2).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=second_hidden_dim, proj_dim=second_hidden_dim).to(device)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs=loss_type).to(device)

    optimizer = Adam(encoder_model.parameters(),
                     lr=args.pre_learning_rate, weight_decay=5e-5)

    with tqdm(total=args.preepochs, desc='(T)') as pbar:
        for epoch in range(args.preepochs):
            loss = train(encoder_model, contrast_model, data, optimizer)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    test_result = test(encoder_model, data)
    with open('./results/nc_FBGCN_Aug_{}_{}.csv'.format(args.dataset, args.loss_type), 'a+') as file:
        file.write('\n')
        file.write('pretrain epochs = {}\n'.format(args.preepochs))
        file.write('pre_learning_rate = {}\n'.format(args.pre_learning_rate))
        file.write('hidden_dim = {}\n'.format(args.hidden_dim))
        file.write('second hidden_dim = {}\n'.format(second_hidden_dim))
        file.write("augmentation ratio = {}\n".format(args.aug))
        file.write("augmentation type = {}\n".format(args.aug_type))
        file.write("augmentation side = {}\n".format(args.aug_side))
        file.write(
            f'(E): FBGCN_Aug: Best test accuracy={test_result["accuracy"]:.4f}')
        file.write('\n')


if __name__ == '__main__':
    main()