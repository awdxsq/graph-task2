import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GCNConv,GATConv,GINConv,GraphSAGEConv

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, g, feature):
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(dgl.function.copy_src(src='h', out='m'), dgl.function.sum(msg='m', out='h'))
            h = g.ndata['h']
            return F.relu(self.linear(h))

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNLayer(input_dim, hidden_dim)
        self.conv2 = GCNLayer(hidden_dim, output_dim)

    def forward(self, g, features):
        x = self.conv1(g, features)
        x = self.conv2(g, x)
        return x


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads, dropout, alpha):
        super(GATLayer, self).__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.alpha = alpha

        self.attentions = nn.ModuleList(
            [GATConv(in_features, out_features, num_heads=1, dropout=dropout, alpha=alpha) for _ in range(num_heads)])

    def forward(self, g, feature):
        out = feature
        for i, attn in enumerate(self.attentions):
            out = attn(g, out)
            if i != 0:
                out = F.dropout(out, self.dropout, training=self.training)
        return F.elu(out)


class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, dropout, alpha):
        super(GAT, self).__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.alpha = alpha

        self.conv1 = GATLayer(input_dim, hidden_dim, num_heads=num_heads, dropout=dropout, alpha=alpha)
        self.conv2 = GATLayer(hidden_dim, output_dim, num_heads=num_heads, dropout=dropout, alpha=alpha)

    def forward(self, g, features):
        x = self.conv1(g, features)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(g, x)
        return x

class GraphSAGELayer(nn.Module):
    def __init__(self, in_features, out_features, num_samples):
        super(GraphSAGELayer, self).__init__()
        self.num_samples = num_samples
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, g, feature):
        out = feature
        for i in range(self.num_samples):
            neighbor_features = dgl.sampling.sample_neighbors(g, g.ndata['id'], num_neighbors=self.num_samples)
            neighbor_features = neighbor_features.ndata['h']
            out = self.linear(torch.cat((out, neighbor_features), dim=1))
        return F.relu(out)

class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_samples):
        super(GraphSAGE, self).__init__()
        self.num_samples = num_samples
        self.conv1 = GraphSAGELayer(input_dim, hidden_dim, num_samples=num_samples)
        self.conv2 = GraphSAGELayer(hidden_dim, output_dim, num_samples=num_samples)

    def forward(self, g, features):
        x = self.conv1(g, features)
        x = F.relu(x)
        x = self.conv2(g, x)
        return x

class GINLayer(nn.Module):
    def __init__(self, in_features, out_features, num_layers):
        super(GINLayer, self).__init__()
        self.num_layers = num_layers
        self.linear = nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(num_layers)])

    def forward(self, g, feature):
        out = feature
        for i, layer in enumerate(self.linear):
            out = layer(out)
            if i != 0:
                out = F.relu(out)
        return out

class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.conv1 = GINConv(GINLayer(input_dim, hidden_dim, num_layers=num_layers), aggregator_type='mean')
        self.conv2 = GINConv(GINLayer(hidden_dim, output_dim, num_layers=num_layers), aggregator_type='mean')

    def forward(self, g, features):
        x = self.conv1(g, features)
        x = F.relu(x)
        x = self.conv2(g, x)
        return x
