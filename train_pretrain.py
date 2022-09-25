import torch
import GCL.losses as L
import torch.nn.functional as F
from tqdm import tqdm
from utility.data import build_graph
from model.downstream import FBGCN, GCN, GAT
from utility.eval import evaluate_metrics, EarlyStopping
import numpy as np
from utility.config import get_arguments
# from model.pretrain import Pre_HighPass, Pre_LowPass, pt_model
from model.pretrain import Pre_Train
from MIX_FBGCN import Encoder
from torch.optim import Adam
from GCL.models import DualBranchContrast
import GCL.augmentors as A
from collections import OrderedDict


@torch.no_grad()
def validate(data, model,r):
    model.eval()
    out = model(data.x, data.lsym, data.anorm)
    return F.nll_loss(out[data.val_mask[r] == 1], data.y[data.val_mask[r] == 1])

@torch.no_grad()
def validate_base(data, model,r):
    model.eval()
    out = model(data.x, data.edge_index)
    return F.nll_loss(out[data.val_mask[r] == 1], data.y[data.val_mask[r] == 1])

@torch.no_grad()
def evaluate_base(model, data,r):
    model.eval()
    out = model(data.x, data.edge_index)

    return evaluate_metrics(data, out,r)

@torch.no_grad()
def evaluate(model, data,r):
    model.eval()
    out = model(data.x, data.lsym, data.anorm)

    return evaluate_metrics(data, out,r)

def train_pre(encoder_model, contrast_model, data, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    z, z1, z2 = encoder_model(data.x, data.edge_index, data.lsym, data.anorm)
    loss = contrast_model(z1, z2)
    loss.backward()
    optimizer.step()
    return loss.item()


def train(data, model, optimizer, r):
    out = model(data.x, data.lsym, data.anorm)
    loss = F.nll_loss(out[data.train_mask[r] == 1], data.y[data.train_mask[r] == 1])
    loss.backward()
    optimizer.step()
    return loss.item()

def train_base(data, model, optimizer, r):
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask[r] == 1], data.y[data.train_mask[r] == 1])
    loss.backward()
    optimizer.step()
    return loss.item()


def main():
    args = get_arguments()
    print(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = build_graph(args.dataset).to(device)
    aug1 = A.FeatureDropout(pf=1.)
    aug2 = A.FeatureDropout(pf=1.)
    with open('./results/nc_{}_{}_{}.csv'.format(args.dataset,args.gnn, args.loss_type), 'a+') as file:
        hidden_dim = args.hidden_dim
        val_acc_list, test_acc_list, train_acc_list = [], [], []       
        for r in range(5):
            if args.preepochs != 0:
                fbconv = Pre_Train(2, data.num_features, hidden_dim, data.num_classes, 0.5)
                if(args.loss_type == "False"):
                    loss_type = False
                else:
                    loss_type = True
                contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs= loss_type).to(device)
                encoder_model = Encoder(pretrain_model=fbconv, hidden_dim=1, proj_dim=1, augmentor=(aug1, aug2)).to(device)
                optimizer = Adam(encoder_model.parameters(), lr=args.pre_learning_rate, weight_decay=5e-5)
                loss = 0
                with tqdm(total=args.preepochs, desc='(T)') as pbar:
                    for epoch in range(args.preepochs):
                        loss = train_pre(encoder_model, contrast_model, data, optimizer)
                        pbar.set_postfix({'loss': loss})
                        pbar.update()
                # file.write('pretrain loss = {}\n'.format(loss))
            early_stopping = EarlyStopping(patience = args.patience)
            if args.gnn == "gcn":
                model = GCN(2,data.num_features, hidden_dim, data.num_classes, 0.5).to(device)
            elif args.gnn == "gat":
                model = GAT(2,data.num_features, 8, data.num_classes, 0.5).to(device)
            else:
                model = FBGCN(2,data.num_features, hidden_dim, data.num_classes, 0.5).to(device)
            model.train()
            if (args.preepochs != 0):
                od = OrderedDict()
                od['gcns.0.lin.weight'] = encoder_model.state_dict()['pretrain_model.stacks.0.encoder.weight']
                od['gcns.1.lin.weight'] = encoder_model.state_dict()['pretrain_model.stacks.1.encoder.weight']
                model.load_state_dict(od, strict = False)
            optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            lowest_val_loss = float("inf")
            best_test, best_val, best_tr = 0, 0, 0  
            for epoch in range(args.epochs):
                if args.gnn == "gcn" or args.gnn == "gat":
                    train_loss = train_base(data, model, optimizer, r)
                    val_loss = validate_base(data, model, r)
                else:
                    train_loss = train(data, model, optimizer, r)
                    val_loss = validate(data, model, r)
                # print(f'Run: {r + 1}, Epoch: {epoch:02d}, Loss: {train_loss:.4f}')
                if lowest_val_loss > val_loss or epoch == args.epochs - 1:
                    lowest_val_loss = val_loss
                    if args.gnn == "gcn"or args.gnn == "gat":
                        evals = evaluate_base(model, data, r)
                    else:
                        evals = evaluate(model, data, r)
                    best_val = evals['val_acc']
                    best_test = evals['test_acc']
                    best_tr = evals['train_acc']
                early_stopping(val_loss, model)
                if early_stopping.early_stop or epoch == args.epochs - 1:
                    print(f'Train acc: {best_tr:.4f}, Validation acc: {best_val:.4f}, Test acc: {best_test:.4f}')
                    val_acc_list.append(best_val)
                    train_acc_list.append(best_tr)
                    test_acc_list.append(best_test)
                    # if r == 9:
                    #     torch.save(model.state_dict(), "saved_model/" + str(name_data))
                    break
        print(f'total,{np.mean(train_acc_list):.4f}, {np.mean(val_acc_list):.4f}, {np.mean(test_acc_list):.4f}\n')
        file.write('\n')
        file.write('pretrain epochs = {}\n'.format(args.preepochs))
        file.write('epochs = {}\n'.format(args.epochs))
        file.write('learning rate = {}\n'.format(args.learning_rate))
        file.write('hidden_dim = {}\n'.format(hidden_dim))
        file.write('pre_learning_rate = {}\n'.format(args.pre_learning_rate))
        file.write('run, train acc avg, validation acc avg, test acc avg\n')
        file.write(f'total {np.mean(train_acc_list):.4f}, {np.mean(val_acc_list):.4f},{np.mean(test_acc_list):.4f}\n')


if __name__ == '__main__':
    main()