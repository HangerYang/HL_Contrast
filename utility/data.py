import torch
import numpy as np
from torch_geometric.utils import to_networkx, add_self_loops
import scipy as sp
import networkx as nx
from torch_geometric.transforms import RandomNodeSplit


def csr_to_sparse(csr):
    csr = csr.tocoo().astype(np.float32)
    indices = torch.from_numpy(
    np.vstack(
        (csr.row, csr.col)).astype(np.int64)
    )
    values = torch.from_numpy(csr.data)
    shape = torch.Size(csr.shape)
    sparse = torch.sparse.FloatTensor(indices, values, shape)
    return sparse

def build_graph(dataset):
    data = dataset[0]
    data.num_classes = dataset.num_classes
    data.edge_index, _ = add_self_loops(data.edge_index)
    data = mask_on(data)
    g = to_networkx(data, to_undirected=True)
    Lsym = nx.linalg.laplacianmatrix.normalized_laplacian_matrix(g)
    Anorm = sp.sparse.identity(np.shape(Lsym)[0]) - Lsym
    adj = nx.adjacency_matrix(g)

    data.adj =csr_to_sparse(adj)
    data.lsym = csr_to_sparse(Lsym)
    data.anorm = csr_to_sparse(Anorm)
    data.degree = np.sum(adj, axis=1)
    return data

# def semi_masks(data, train_ratio = 0.1, val_ratio = 0.1, test_ratio = 0.8):
    # num_class = data.num_classes
    # num_nodes = data.num_nodes
    # train_mask = torch.zeros(num_nodes)
    # test_mask = torch.zeros(num_nodes)
    # val_mask = torch.zeros(num_nodes)

    # val_size = int(num_nodes*val_ratio)
    # test_size = int(num_nodes*test_ratio)
    # # 5% of the nodes would be labelled, +1 for int truncation
    # train_size = int(num_nodes*train_ratio/num_class)+1
    # for i in range(num_class):
    #     # each class would have a fixed number of nodes being labelled
    #     train_index = (data.y == i).nonzero(as_tuple = True)[0]
    #     # the selected index would be marked as True
    #     selected_train_index = torch.randperm(n = len(train_index))
    #     for j in selected_train_index[:train_size]:
    #         train_mask[train_index[j]] = True
    
    # excluded_idx_train = train_mask.nonzero(as_tuple=True)[0]
    # temp = [i for i in range(num_nodes) if i not in excluded_idx_train]
    # selected_val_index = torch.randperm(n = len(temp))
    # for i in selected_val_index[:val_size]:
    #     val_mask[temp[i]] = True
    # excluded_idx_val = val_mask.nonzero(as_tuple=True)[0]
    # temp = [i for i in range(num_nodes) if i not in excluded_idx_train and i not in excluded_idx_val]
    # selected_test_index = torch.randperm(n = len(temp))
    # for i in selected_test_index[:test_size]:
    #     test_mask[temp[i]] = True
    #     data.train_mask, data.val_mask, data.test_mask = (train_mask, val_mask, test_mask) 
    # return data
def semi_masks(data):
    transform = RandomNodeSplit(
        "test_rest", 
        num_train_per_class=int(data.num_nodes/data.num_classes*0.1), 
        num_val=int(data.num_nodes/data.num_classes*0.1)
        )
    data = transform(data)
    return data
def  mask_on(data):
    data.train_mask = np.swapaxes(data.train_mask, 0, 1)
    data.val_mask = np.swapaxes(data.val_mask, 0, 1)
    data.test_mask = np.swapaxes(data.test_mask, 0, 1)
    return data