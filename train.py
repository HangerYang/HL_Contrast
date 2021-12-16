import torch
import GCL.losses as L
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam
from GCL.models import DualBranchContrast
from torch_geometric.datasets import Planetoid, WebKB, Actor, WikipediaNetwork
from utility.data import build_graph
from model.downstream import FBGCN, GCN, GAT
from utility.eval import evaluate_metrics, EarlyStopping
import numpy as np
from model.pretrain import Pre_HighPass, Pre_LowPass, pt_model


@torch.no_grad()
def validate(data, model, r):
    model.eval()
    out = model(data.x, data.lsym, data.anorm)
    return F.nll_loss(out[data.val_mask[r] == 1], data.y[data.val_mask[r] == 1])

@torch.no_grad()
def validate_base(data, model, r):
    model.eval()
    out = model(data.x, data.edge_index)
    return F.nll_loss(out[data.val_mask[r] == 1], data.y[data.val_mask[r] == 1])

@torch.no_grad()
def evaluate_base(model, data, r):
    model.eval()
    out = model(data.x, data.edge_index)

    return evaluate_metrics(data, out, r)

@torch.no_grad()
def evaluate(model, data, r):
    model.eval()
    out = model(data.x, data.lsym, data.anorm)

    return evaluate_metrics(data, out, r)

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
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    epochs = 2000
    # name_model = "GCN"
    # name_model = "GAT"
    name_model = "FBGCN"
    intraview_negs = True
    mes = ""
    if intraview_negs:
        mes = "_intra"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # name_data = 'Actor'
    # dataset = Actor(root = './data/' + "Actor")

    # name_data = 'Cora'
    # dataset = Planetoid(root= './data/' + name_data, name = name_data)

    # name_data = 'Cornell'
    # dataset = WebKB(root= './data/' + name_data, name = name_data)

    # name_data = 'Chameleon'
    # dataset = WikipediaNetwork(root= './data/' + name_data, name = name_data)
    
    # name_data = 'Squirrel'
    # dataset = WikipediaNetwork(root= './data/' + name_data, name = name_data)
    
    name_data = 'Wisconsin'
    dataset = WebKB(root= './data/' + name_data, name = name_data)
        
    # name_data = 'Texas'
    # dataset = WebKB(root= './data/' + name_data, name = name_data)

    data = build_graph(dataset).to(device)
    # pretrain_epochs = 100
    with open('./results/nc_{}_{}{}.csv'.format(name_data, name_model, mes), 'a+') as file:
        file.write('')
        if name_model == "GCN" or name_model == "GAT":
            k = [0]
        else:
            k = [600]
        for pretrain_epochs in k:
            file.write('pretrain epochs = {}\n'.format(pretrain_epochs))
            if pretrain_epochs != 0:
                high_model = Pre_HighPass(2, data.num_features, 64, dataset.num_classes, 0.5).to(device)
                low_model = Pre_LowPass(2, data.num_features, 64, dataset.num_classes, 0.5).to(device)

                contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs=intraview_negs).to(device)
                parameter = list(high_model.parameters()) + list(low_model.parameters())
                optimizer = Adam(parameter, lr=0.001, weight_decay=5e-5)

                with tqdm(total=pretrain_epochs, desc='(T)') as pbar:
                    for epoch in range(pretrain_epochs):
                        loss = pt_model(high_model, low_model, contrast_model, optimizer, data)
                        pbar.set_postfix({'loss': loss})
                        pbar.update()
                file.write('pretrain loss = {}\n'.format(loss))
                # torch.save(high_model.state_dict(), './pretrain_param/high_model.pth')
                # torch.save(low_model.state_dict(), './pretrain_param/low_model.pth')
            val_acc_list, test_acc_list, train_acc_list = [], [], []
            
            for r in range(10):
                early_stopping = EarlyStopping(patience = 200)
                if name_model == "GCN":
                    model = GCN(2,data.num_features, 64, dataset.num_classes, 0.5).to(device)
                elif name_model == "GAT":
                    model = GAT(2,data.num_features, 8, dataset.num_classes, 0.5).to(device)
                else:
                    model = FBGCN(2,data.num_features, 64, dataset.num_classes, 0.5).to(device)
                model.train()
                if (pretrain_epochs != 0):
                    model.load_state_dict(high_model.state_dict(), strict = False)
                    model.load_state_dict(low_model.state_dict(), strict = False)
                optimizer = Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
                lowest_val_loss = float("inf")
                best_test, best_val, best_tr = 0, 0, 0  
                for epoch in range(epochs):
                    if name_model == "GCN" or name_model == "GAT":
                        train_loss = train_base(data, model, optimizer, r)
                        val_loss = validate_base(data, model, r)
                    else:
                        train_loss = train(data, model, optimizer, r)
                        val_loss = validate(data, model, r)
                    # print(f'Run: {r + 1}, Epoch: {epoch:02d}, Loss: {train_loss:.4f}')
                    if lowest_val_loss > val_loss or epoch == epochs - 1:
                        lowest_val_loss = val_loss
                        if name_model == "GCN"or name_model == "GAT":
                            evals = evaluate_base(model, data, r)
                        else:
                            evals = evaluate(model, data, r)
                        best_val = evals['val_acc']
                        best_test = evals['test_acc']
                        best_tr = evals['train_acc']
                    early_stopping(val_loss, model)
                    if early_stopping.early_stop or epoch == epochs - 1:
                        print(f'Train acc: {best_tr:.4f}, Validation acc: {best_val:.4f}, Test acc: {best_test:.4f}')
                        val_acc_list.append(best_val)
                        train_acc_list.append(best_tr)
                        test_acc_list.append(best_test)
                        # if r == 9:
                        #     torch.save(model.state_dict(), "saved_model/" + str(name_data))
                        break
            print(f'total,{np.mean(train_acc_list):.4f}, {np.mean(val_acc_list):.4f}, {np.mean(test_acc_list):.4f}\n')
            file.write('run, train acc avg, validation acc avg, test acc avg\n')
            file.write(f'total,{np.mean(train_acc_list):.4f}, {np.mean(val_acc_list):.4f},{np.mean(test_acc_list):.4f}\n')


if __name__ == '__main__':
    main()