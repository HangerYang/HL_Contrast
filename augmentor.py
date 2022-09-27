from torch_geometric.utils import sort_edge_index
import torch
from torch_sparse import coalesce
from GCL.augmentors.augmentor import Graph, Augmentor

def coalesce_edge_index(edge_index, edge_weights=None):
    num_edges = edge_index.size()[1]
    num_nodes = edge_index.max().item() + 1
    edge_weights = edge_weights if edge_weights is not None else torch.ones((num_edges,), dtype=torch.float32, device=edge_index.device)

    return coalesce(edge_index, edge_weights, m=num_nodes, n=num_nodes)
def add_edge(edge_index: torch.Tensor, ratio: float) -> torch.Tensor:
    num_edges = edge_index.size()[1]
    num_nodes = edge_index.max().item() + 1
    num_add = int(num_edges * ratio)

    new_edge_index = torch.randint(0, num_nodes - 1, size=(2, num_add)).to(edge_index.device)
    edge_index = torch.cat([edge_index, new_edge_index], dim=1)

    edge_index = sort_edge_index(edge_index)

    return coalesce_edge_index(edge_index)[0]


class EdgeAdding(Augmentor):
    def __init__(self, pe: float):
        super(EdgeAdding, self).__init__()
        self.pe = pe

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        edge_index = add_edge(edge_index, ratio=self.pe)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)