# Author: Piyush Vyas
import torch
from layers import GraphConv


class GCN(torch.nn.Module):
    def __init__(self, fan_in, fan_out, layers, dropout=0.2, normalize=True, bias=False, precalc=False):
        super(GCN, self).__init__()
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.num_layers = layers
        self.dropout = dropout
        self.normalize = normalize
        self.bias = bias
        self.precalc = precalc
        self.network = torch.nn.ModuleList([])
        self.network.append(
            GraphConv(in_features=fan_in[0], out_features=fan_out[0], dropout=dropout, bias=bias,
                      normalize=normalize, precalc=self.precalc))
        for i in range(1, layers-1):
            self.network.append(
                GraphConv(in_features=fan_in[i], out_features=fan_out[i], dropout=dropout, bias=bias,
                          normalize=normalize, precalc=False))
        self.network.append(
            GraphConv(in_features=fan_in[-1], out_features=fan_out[-1], dropout=dropout, bias=self.bias,
                      normalize=False, last=True, precalc=False))

    def forward(self, sparse_adj, feats):
        for _, layer in enumerate(self.network):
            feats = layer(sparse_adj, feats)
        return feats
