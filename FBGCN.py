from torch import cat
from torch.optim import Adam
from GCL.models import DualBranchContrast
from GCL.eval import get_split
import GCL.losses as L
from tqdm import tqdm
from Evaluator import LREvaluator
import torch
from utility.config import get_arguments

# sys.path.append(os.path.abspath(os.path.join('..', 'model')))
from model.pretrain import Pre_HighPass, Pre_LowPass, pt_model
# sys.path.append(os.path.abspath(os.path.join('..', 'utility')))
from utility.data import build_graph

def test(high_pass_model, low_pass_model, data):
    high_pass_model.eval()
    low_pass_model.eval()
    z1 = high_pass_model(data.x, data.lsym)
    # print(z1.dim)
    z2 = low_pass_model(data.x, data.anorm)
    # z = cat((z1, z2), 1)
    z = z2
    split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = LREvaluator()(z, data.y, split)
    return result


def main():
    args = get_arguments()
    print(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = build_graph(args.dataset).to(device)
    hidden_dim = args.hidden_dim

    second_hidden_dim = int(hidden_dim/2)
    if(args.loss_type == "False"):
        loss_type = False
    else:
        loss_type = True
    pre_learning_rate = args.pre_learning_rate
    high_model = Pre_HighPass(2, data.num_features, hidden_dim, second_hidden_dim, 0.5).to(device)
    low_model = Pre_LowPass(2, data.num_features, hidden_dim, second_hidden_dim, 0.5).to(device)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs= loss_type).to(device)
    parameter = list(high_model.parameters()) + list(low_model.parameters())
    # optimizer = Adam(parameter, lr=pre_learning_rate, weight_decay=5e-5)
    optimizer = Adam(low_model.parameters(), lr=pre_learning_rate, weight_decay=5e-5)

    with tqdm(total=args.preepochs, desc='(T)') as pbar:
        for epoch in range(args.preepochs):
            loss = pt_model(low_model, high_model, contrast_model, optimizer, data)
            pbar.set_postfix({'loss': loss})
            pbar.update()
    test_result = test(low_model, high_model, data)
    with open('./results/nc_FBGCN_{}_{}.csv'.format(args.dataset, args.loss_type), 'a+') as file:
        file.write('\n')
        file.write('pretrain epochs = {}\n'.format(args.preepochs))
        file.write('pre_learning_rate = {}\n'.format(args.pre_learning_rate))
        file.write('hidden_dim = {}\n'.format(args.hidden_dim))
        file.write('second hidden_dim = {}\n'.format(second_hidden_dim))
        file.write(f'(E): FBGCN, low: Best test accuracy={test_result["accuracy"]:.4f}')


if __name__ == '__main__':
    main()