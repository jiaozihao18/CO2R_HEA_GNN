import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.nn.models.schnet import GaussianSmearing
from .onehot_embeddings import ONEHOT_EMBEDDINGS

class CGCNN(nn.Module):
    """
    atom_fea_len: int
      Number of hidden atom features in the convolutional layers
    n_conv: int
      Number of convolutional layers
    fc_fea_len: int
      Number of hidden features after pooling
    n_fc: int
      Number of hidden layers after pooling
    """

    def __init__(
        self,
        atom_fea_len: int = 64,
        n_conv: int = 3,
        fc_fea_len: int = 128,
        n_fc: int = 1,
        num_gaussians: int = 50,
        cutoff: int = 6,
    ) -> None:
       
        super().__init__()
        embeddings = ONEHOT_EMBEDDINGS
        self.embedding = torch.zeros(100, len(embeddings[1]))
        
        for i in range(100):
            self.embedding[i] = torch.tensor(embeddings[i + 1])
        self.embedding_fc = nn.Linear(len(embeddings[1]), atom_fea_len)

        self.convs = nn.ModuleList([CGCNNConv(node_dim=atom_fea_len, edge_dim=num_gaussians) for _ in range(n_conv)])
        self.conv_to_fc = nn.Sequential(nn.Linear(atom_fea_len, fc_fea_len), nn.Softplus())

        if n_fc > 1:
            layers = []
            for _ in range(n_fc - 1):
                layers.append(nn.Linear(fc_fea_len, fc_fea_len))
                layers.append(nn.Softplus())
            self.fcs = nn.Sequential(*layers)
            
        self.fc_out = nn.Linear(fc_fea_len, 1)

        self.cutoff = cutoff
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

    def forward(self, data):
        # Get node features
        if self.embedding.device != data.atomic_numbers.device:
            self.embedding = self.embedding.to(data.atomic_numbers.device)
        data.x = self.embedding[data.atomic_numbers.long() - 1]

        data.edge_attr = self.distance_expansion(data.distances)
        # Forward pass through the network
        mol_feats = self._convolve(data)
        mol_feats = self.conv_to_fc(mol_feats)
        if hasattr(self, "fcs"):
            mol_feats = self.fcs(mol_feats)

        out = self.fc_out(mol_feats)
        return out

    def _convolve(self, data):
        """
        Returns the output of the convolution layers before they are passed
        into the dense layers.
        """
        node_feats = self.embedding_fc(data.x)
        for f in self.convs:
            node_feats = f(node_feats, data.edge_index, data.edge_attr)
        mol_feats = global_mean_pool(node_feats, data.batch)
        return mol_feats
    
    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class CGCNNConv(MessagePassing):
    """Implements the message passing layer from
    `"Crystal Graph Convolutional Neural Networks for an
    Accurate and Interpretable Prediction of Material Properties"
    <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301>`.
    """

    def __init__(
        self, node_dim, edge_dim, **kwargs
    ) -> None:
        super(CGCNNConv, self).__init__(aggr="add")
        self.node_feat_size = node_dim
        self.edge_feat_size = edge_dim

        self.lin1 = nn.Linear(
            2 * self.node_feat_size + self.edge_feat_size,
            2 * self.node_feat_size,
        )
        self.bn1 = nn.BatchNorm1d(2 * self.node_feat_size)
        self.ln1 = nn.LayerNorm(self.node_feat_size)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.lin1.weight)

        self.lin1.bias.data.fill_(0)

        self.bn1.reset_parameters()
        self.ln1.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        """
        Arguments:
            x has shape [num_nodes, node_feat_size]
            edge_index has shape [2, num_edges]
            edge_attr is [num_edges, edge_feat_size]
        """
        out = self.propagate(
            edge_index, x=x, edge_attr=edge_attr, size=(x.size(0), x.size(0))
        )
        out = nn.Softplus()(self.ln1(out) + x)
        return out

    def message(self, x_i, x_j, edge_attr):
        """
        Arguments:
            x_i has shape [num_edges, node_feat_size]
            x_j has shape [num_edges, node_feat_size]
            edge_attr has shape [num_edges, edge_feat_size]

        Returns:
            tensor of shape [num_edges, node_feat_size]
        """
        z = self.lin1(torch.cat([x_i, x_j, edge_attr], dim=1))
        z = self.bn1(z)
        z1, z2 = z.chunk(2, dim=1)
        z1 = nn.Sigmoid()(z1)
        z2 = nn.Softplus()(z2)
        return z1 * z2
