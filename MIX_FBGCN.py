import torch
from torch import cat
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F
from model.pretrain import Pre_Mix_Layer, Pre_Train

from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split
from Evaluator import LREvaluator
from GCL.models import DualBranchContrast
from utility.data import build_graph
from utility.config import get_arguments


class Encoder(torch.nn.Module):
    def __init__(self, pretrain_model, hidden_dim, proj_dim, augmentor=None):
        super(Encoder, self).__init__()
        self.pretrain_model = pretrain_model
        self.augmentor = augmentor

        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, hidden_dim)

    def forward(self, x, edge_index, lsym, anorm):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, None)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, None)
        z1 = self.pretrain_model(x1, lsym, anorm, "high")
        z2 = self.pretrain_model(x2, lsym, anorm, "low")
        z = self.pretrain_model(x, lsym, anorm, "low")
        # z_2 = self.pretrain_model(x, lsym, anorm, "high")
        # z = cat((z_1, z_2), 1)
        # z = z2
        return z, z1, z2

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)


def train(encoder_model, contrast_model, data, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    z, z1, z2 = encoder_model(data.x, data.edge_index, data.lsym, data.anorm)
    h1, h2 = [encoder_model.project(x) for x in [z1, z2]]
    loss = contrast_model(h1, h2)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(encoder_model, data):
    encoder_model.eval()
    z, _, _ = encoder_model(data.x, data.edge_index, data.lsym, data.anorm)
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
    dataset = args.dataset
    if(args.loss_type == "False"):
        loss_type = False
    else:
        loss_type = True
    data = build_graph(dataset).to(device)
    aug1 = A.FeatureDropout(pf=0.25)
    aug2 = A.FeatureDropout(pf=0.25)
    fbconv = Pre_Train(2, data.num_features, hidden_dim,
                       second_hidden_dim, 0.5)
    encoder_model = Encoder(pretrain_model=fbconv, hidden_dim=128,
                            proj_dim=128, augmentor=(aug1, aug2)).to(device)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(
        tau=0.2), mode='L2L', intraview_negs=loss_type).to(device)

    optimizer = Adam(encoder_model.parameters(),
                     lr=args.pre_learning_rate, weight_decay=5e-5)

    with tqdm(total=args.preepochs, desc='(T)') as pbar:
        for epoch in range(args.preepochs):
            loss = train(encoder_model, contrast_model, data, optimizer)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    test_result = test(encoder_model, data)
    with open('./results/nc_FBGCN_{}_{}.csv'.format(args.dataset, args.loss_type), 'a+') as file:
        file.write('\n')
        file.write('pretrain epochs = {}\n'.format(args.preepochs))
        file.write('pre_learning_rate = {}\n'.format(args.pre_learning_rate))
        file.write('hidden_dim = {}\n'.format(args.hidden_dim))
        file.write('second hidden_dim = {}\n'.format(second_hidden_dim))
        file.write(
            f'(E): FBGCN: Best test accuracy={test_result["accuracy"]:.4f}')


if __name__ == '__main__':
    main()