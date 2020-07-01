import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Parameter
from tqdm import tqdm
import torch_geometric
from torch_geometric.datasets import Reddit
from torch_geometric.data import ClusterData, ClusterLoader, Data, NeighborSampler
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_scatter import scatter_add

dataset = Reddit('../data/Reddit')
data = dataset[0]

# Separate train, val, test
train_data = Data()
train_data.x = data.x[data.train_mask]
train_data.y = data.y[data.train_mask]
train_data.edge_index = torch_geometric.utils.subgraph(data.train_mask, data.edge_index, relabel_nodes=True)[0]
print(train_data)
val_data = Data()
val_data.x = data.x[data.val_mask]
val_data.y = data.y[data.val_mask]
val_data.edge_index = torch_geometric.utils.subgraph(data.val_mask, data.edge_index, relabel_nodes=True)[0]
val_data.mask = data.val_mask
print(val_data)
test_data = Data()
test_data.x = data.x[data.test_mask]
test_data.y = data.y[data.test_mask]
test_data.edge_index = torch_geometric.utils.subgraph(data.test_mask, data.edge_index, relabel_nodes=True)[0]
test_data.mask = data.test_mask
print(test_data)

train_data = ClusterData(
    train_data, num_parts=1500, recursive=False,
    save_dir="data/Reddit/train")
val_data = ClusterData(
    val_data, num_parts=20, recursive=False,
    save_dir="data/Reddit/val")
test_data = ClusterData(
    test_data, num_parts=1, recursive=False,
    save_dir="data/Reddit/test")

train_loader = ClusterLoader(
    train_data, batch_size=20, shuffle=True, num_workers=8)
val_loader = ClusterLoader(
    val_data, batch_size=1, shuffle=False, num_workers=8)
test_loader = ClusterLoader(
    test_data, batch_size=1, shuffle=False, num_workers=1)

# subgraph_loader = NeighborSampler(data.edge_index, sizes=[-1], batch_size=1024,
#                                   shuffle=False, num_workers=12)


class GCNConvDiagEnhance(MessagePassing):
    def __init__(self, in_channels, out_channels, improved=False, cached=False,
                 bias=True, normalize=True, **kwargs):
        super(GCNConvDiagEnhance, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def diag_enhance_norm(edge_index, num_nodes, edge_weight=None, improved=False,
                          dtype=None, diag_lambda=1.0):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm_edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        diag_edge_weight = norm_edge_weight.clone()
        diag_edge_weight[edge_index[0] != edge_index[1]] = 0

        return (edge_index, norm_edge_weight + diag_lambda * diag_edge_weight)

    def forward(self, x, edge_index, edge_weight=None, diag_lambda=1.0):
        """"""
        edge_index, norm = self.diag_enhance_norm(
            edge_index, x.size(self.node_dim), edge_weight, self.improved,
            x.dtype, diag_lambda)
        support = self.propagate(edge_index, x=x, norm=norm)
        support = torch.cat((support, x), dim=1)
        support = F.dropout(support, p=0.2)
        out = torch.matmul(support, self.weight)
        out = F.layer_norm(out, [self.out_channels])
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j if norm is not None else x_j

    def update(self, aggr_out):
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, f_size=128,
                 diag_lambda=0.0001):
        super(Net, self).__init__()
        self.convs = ModuleList(
            [GCNConvDiagEnhance(2*in_channels, f_size, bias=False),
            #  GCNConvDiagEnhance(2*f_size, f_size, bias=False),
            #  GCNConvDiagEnhance(2*f_size, f_size, bias=False),
             GCNConvDiagEnhance(2*f_size, out_channels, bias=False)])
        self.diag_lambda = diag_lambda
        self.in_fea = [in_channels, f_size, f_size, f_size]
        self.out_fea = [f_size, f_size, f_size, out_channels]
        self.f_size = f_size

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, diag_lambda=self.diag_lambda)
            if i != len(self.convs) - 1:
                x = F.relu(x)
        return x

    def inference(self, x_all):
        pbar = tqdm(total=x_all.size(0) * len(self.convs))
        pbar.set_description('Evaluating')
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i, conv in enumerate(self.convs):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x = conv(x, edge_index, diag_lambda=self.diag_lambda)
                if i != len(self.convs) - 1:
                    x = F.relu(x)
                xs.append(x.cpu())
                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu_device = torch.device('cpu')
model = Net(dataset.num_features, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)


def train():
    model.train()

    total_loss = total_nodes = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = F.cross_entropy(out, batch.y)
        loss.backward()
        optimizer.step()

        nodes = batch.x.size(0)
        total_loss += loss.item() * nodes
        total_nodes += nodes

    return total_loss / total_nodes


@torch.no_grad()
def evaluate(loader, device):  # Inference should be performed on the full graph.
    model.eval()

    total_loss = total_nodes = total_correct = 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        loss = F.cross_entropy(out, batch.y).detach()
        y_pred = out.argmax(dim=-1)
        correct = y_pred.eq(batch.y).sum().item()

        nodes = batch.x.size(0)
        total_loss += loss.item() * nodes
        total_nodes += nodes
        total_correct += correct
    return total_loss / total_nodes, total_correct / total_nodes


@torch.no_grad()
def test():  # Inference should be performed on the full graph.
    model.eval()
    out = model.inference(data.x)
    y_pred = out.argmax(dim=-1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = y_pred[mask].eq(data.y[mask]).sum().item()
        accs.append(correct / mask.sum().item())
    return accs


for epoch in range(1, 130+1):
    loss = train()
    if epoch % 5 == 0:
        with torch.no_grad():
            val_loss, val_acc = evaluate(val_loader, device)
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, '
                  f'Val: loss:{val_loss:.4f}, acc:{val_acc:.4f}')
            # train_acc, val_acc, test_acc = test()
            # print(train_acc, val_acc, test_acc)
    else:
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
# Test
# use cpu for inference
# model.to(cpu_device)
# test_loss, test_acc = evaluate(test_loader, cpu_device)
# print(f'Epoch: {epoch:02d}, '
#       f'Test: loss:{test_loss:.4f}, acc:{test_acc:.4f}')
