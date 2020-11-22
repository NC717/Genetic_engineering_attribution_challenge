'''
# `All GNN architectures tried during the competiton`
- `Graph Attention Networks`
- `Graph convolution neural networks`
- `Sort Pool`
'''

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GATConv
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_sort_pool
from torch.nn import Linear, Conv1d



in_channels = 14
output_dim = 4
hidden_channels = 128

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, output_dim ):
        super(GCN, self).__init__()
        torch.manual_seed(12345)

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, output_dim)

    def forward(self, x, edge_index, batch):
        # X- Node features
        # Edge matrix 
        # Batch size

        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin(x)
        return x


class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, output_dim):
        super(Net, self).__init__()

        self.conv1 = nn.Sequential(GATConv(in_channels, hidden_channels, heads = 8, dropout=0.2), nn.ReLU())
        self.conv2 = nn.Sequential(GATConv(hidden_channels, hidden_channels, heads = 8, dropout=0.2), nn.ReLU())
        # On the Pubmed dataset, use heads=8 in conv2.

        self.conv3 = nn.Sequential(GATConv(hidden_channels * hidden_channels, output_dim, heads=1, concat=False, dropout=0.6), nn.ReLU())
        self.Dropout = nn.Dropout(p = 0.5)

    def forward(self, x, edge_index):
        x = self.Dropout(x)
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x)

        print(x.shape)
        return x

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, output_dim):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(hidden_channels * 8, output_dim, heads=1, concat=False,
                             dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x , p=0.6, training=self.training)
        print(x.shape, edge_index.shape)

        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        print(x.shape, edge_index.shape)
        x = self.conv2(x,  edge_index)
        print(x.shape)

        return x

class SortPool(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, output_dim, num_layers):
        super(SortPool, self).__init__()
        self.k = 30
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.conv1d = Conv1d(hidden_channels, 32, 5)
        self.lin1 = Linear(32 * (self.k - 5 + 1), hidden_channels)
        self.lin2 = Linear(hidden_channels, output_dim)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.conv1d.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index, batch):
        # x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))

        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_sort_pool(x, batch, self.k)
        x = x.view(len(x), self.k, -1).permute(0, 2, 1)
        x = F.relu(self.conv1d(x))
        x = x.view(len(x), -1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x