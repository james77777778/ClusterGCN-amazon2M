# Author: Piyush Vyas
import os
import time
import tqdm
import torch
import argparse
import numpy as np
import utils as utils
from models import GCN
from torch.utils.tensorboard import SummaryWriter


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='amazon2M', help='options - amazon2M')
parser.add_argument('--run_name', type=str, default='sota', help='experiment runname')
parser.add_argument('--exp_num', type=str, help='experiment number for tensorboard')
parser.add_argument('--test', type=int, default=1, help='True if 1, else False')
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--num_clusters_train', type=int, default=10000)
parser.add_argument('--num_clusters_val', type=int, default=20)
parser.add_argument('--num_clusters_test', type=int, default=1)
parser.add_argument('--layers', type=int, default=4, help='Number of layers in the network.')
parser.add_argument('--epochs', type=int, default=2048, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=200, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (ratio of units to drop).')
parser.add_argument('--multilabel', type=bool, default=False, help='Whether a multilabel problem.')
parser.add_argument('--cache', type=bool, default=False, help='Whether to use partition cache.')
parser.add_argument('--diag_lambda', type=float, default=0.0001, help='diag_lambda.')
args = parser.parse_args()


# Reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Tensorboard Writer
writer = SummaryWriter('runs/' + args.dataset + '/' + args.exp_num)

# Settings based on ClusterGCN's Table 4. of section Experiments
if args.dataset == 'amazon2M':
    _in = [100]
    _out = []
    for _ in range(args.layers-1):
        _in.append(args.hidden)
        _out.append(args.hidden)
    _out.append(47)

if args.dataset == 'ppi':
    _in = [50]
    _out = []
    for _ in range(args.layers-1):
        _in.append(args.hidden)
        _out.append(args.hidden)
    _out.append(121)

if args.dataset == 'reddit' and args.layers == 4:
    print("Reddit")
    _in = [602, args.hidden, args.hidden, args.hidden, args.hidden]
    _out = [args.hidden, args.hidden, args.hidden, args.hidden, 41]


def train(model, criterion, optimizer, features, adj, labels, mask, device):
    optimizer.zero_grad()

    features = torch.from_numpy(features).to(device)
    labels = torch.LongTensor(labels).to(device)

    # Adj -> Torch Sparse Tensor
    i = torch.LongTensor(adj[0])  # indices
    v = torch.FloatTensor(adj[1])  # values
    adj = torch.sparse.FloatTensor(i.t(), v, adj[2]).to(device)
    model.to(device)
    output = model(adj, features)
    if args.multilabel:
        loss = criterion(output, labels.type_as(output))
        pred = output.detach()
        pred[pred > 0] = 1
        pred[pred <= 0] = 0
        labels = labels
    else:
        loss = criterion(output, torch.max(labels, 1)[1])
        pred = output.argmax(dim=1, keepdim=True).squeeze().detach()
        labels = torch.max(labels, 1)[1].squeeze()

    loss.backward()
    optimizer.step()

    return loss.item(), pred, labels


@torch.no_grad()
def evaluate(model, criterion, features_batches, support_batches, labels_batches, mask_batches, nodes, device):
    # Evaluate model
    total_pred = []
    total_lab = []
    total_loss = 0
    total_acc = 0
    total_nodes = 0

    num_batches = len(features_batches)
    for i in range(num_batches):
        features_b = features_batches[i]
        support_b = support_batches[i]
        label_b = labels_batches[i]
        mask_b = mask_batches[i]
        num_data_b = np.sum(mask_b)
        if num_data_b == 0:
            continue
        else:
            # evaluate function
            features = torch.from_numpy(features_b).to(device)
            labels = torch.LongTensor(label_b).to(device)
            i = torch.LongTensor(support_b[0])  # indices
            v = torch.FloatTensor(support_b[1])  # values
            adj = torch.sparse.FloatTensor(i.t(), v, support_b[2]).to(device)
            model.to(device)
            output = model(adj, features)
            if args.multilabel:
                loss = criterion(output[mask_b], labels[mask_b].type_as(output))
                pred = output[mask_b]
                pred[pred > 0] = 1
                pred[pred <= 0] = 0
                labels = labels[mask_b]
                total_acc += torch.eq(pred, labels).all(dim=1).sum().item()
            else:
                loss = criterion(output[mask_b], torch.max(labels[mask_b], 1)[1])
                pred = output[mask_b].argmax(dim=1, keepdim=True)
                labels = torch.max(labels[mask_b], 1)[1]
                total_acc += torch.eq(pred.squeeze(), labels.squeeze()).sum().item()
            total_nodes += num_data_b
        total_pred.append(pred)
        total_lab.append(labels)
        total_loss += loss.item()

    total_pred = torch.cat(total_pred).cpu().squeeze().numpy()
    total_lab = torch.cat(total_lab).cpu().squeeze().numpy()
    loss = total_loss / num_batches
    acc = total_acc / total_nodes
    micro, macro = utils.calc_f1(total_pred, total_lab, args.multilabel)
    return loss, acc, micro, macro


@torch.no_grad()
def test(model, criterion, features, adj, labels, mask, device):
    features = torch.FloatTensor(features).to(device)
    labels = torch.LongTensor(labels).to(device)
    total_correct = 0
    if device == torch.device("cpu"):
        adj = adj[0]
        features = features[0]
        labels = labels[0]
        mask = mask[0]

    # Adj -> Torch Sparse Tensor
    i = torch.LongTensor(adj[0])  # indices
    v = torch.FloatTensor(adj[1])  # values
    adj = torch.sparse.FloatTensor(i.t(), v, adj[2]).to(device)
    model.to(device)
    output = model(adj, features)
    if args.multilabel:
        loss = criterion(output, labels.type_as(output))
        pred = torch.sigmoid(output) >= 0.5
        total_correct += torch.eq(pred.squeeze(), labels.squeeze()).all(dim=1).sum().item()
    else:
        loss = criterion(output, torch.max(labels, 1)[1])
        pred = output[mask].argmax(dim=1, keepdim=True)
        labels = torch.max(labels[mask], 1)[1]
        total_correct += torch.eq(pred.squeeze(), labels.squeeze()).sum().item()
    acc = total_correct / sum(mask)
    micro, macro = utils.calc_f1(pred.squeeze(), labels.squeeze(), args.multilabel)
    return loss.item(), acc, micro, macro


def main():
    # Make dir
    temp = "./tmp"
    os.makedirs(temp, exist_ok=True)
    # Load data
    start = time.time()
    (train_adj, full_adj, train_feats, test_feats, y_train, y_val, y_test,
     train_mask, val_mask, test_mask, _, val_nodes, test_nodes,
     num_data, visible_data) = utils.load_data(args.dataset)
    print('Loaded data in {:.2f} seconds!'.format(time.time() - start))

    start = time.time()
    # Prepare Train Data
    if args.batch_size > 1:
        start = time.time()
        _, parts = utils.partition_graph(train_adj, visible_data, args.num_clusters_train)
        print('Partition graph in {:.2f} seconds!'.format(time.time() - start))
        parts = [np.array(pt) for pt in parts]
    else:
        start = time.time()
        (parts, features_batches, support_batches, y_train_batches, train_mask_batches) = utils.preprocess(
            train_adj, train_feats, y_train, train_mask, visible_data, args.num_clusters_train, diag_lambda=args.diag_lambda)
        print('Partition graph in {:.2f} seconds!'.format(time.time() - start))

    # Prepare valid Data
    start = time.time()
    (_, val_features_batches, val_support_batches, y_val_batches, val_mask_batches) = utils.preprocess(
        full_adj, test_feats, y_val, val_mask, np.arange(num_data), args.num_clusters_val, diag_lambda=args.diag_lambda)
    print('Partition graph in {:.2f} seconds!'.format(time.time() - start))

    # Prepare Test Data
    start = time.time()
    (_, test_features_batches, test_support_batches, y_test_batches, test_mask_batches) = utils.preprocess(
        full_adj, test_feats, y_test, test_mask, np.arange(num_data), args.num_clusters_test, diag_lambda=args.diag_lambda)
    print('Partition graph in {:.2f} seconds!'.format(time.time() - start))

    idx_parts = list(range(len(parts)))

    # model
    model = GCN(
        fan_in=_in, fan_out=_out, layers=args.layers, dropout=args.dropout, normalize=True, bias=False, precalc=True).float()
    model.to(torch.device('cuda'))
    print(model)

    # Loss Function
    if args.multilabel:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # Optimization Algorithm
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Learning Rate Schedule
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer, max_lr=args.lr, steps_per_epoch=int(args.num_clusters_train/args.batch_size), epochs=args.epochs+1,
    #     anneal_strategy='linear')

    pbar = tqdm.tqdm(total=args.epochs, dynamic_ncols=True)
    for epoch in range(args.epochs + 1):
        # Train
        np.random.shuffle(idx_parts)
        start = time.time()
        avg_loss = 0
        total_correct = 0
        n_nodes = 0
        if args.batch_size > 1:
            (features_batches, support_batches, y_train_batches, train_mask_batches) = utils.preprocess_multicluster(
                train_adj, parts, train_feats, y_train, train_mask,
                args.num_clusters_train, args.batch_size, args.diag_lambda)
            for pid in range(len(features_batches)):
                # Use preprocessed batch data
                features_b = features_batches[pid]
                support_b = support_batches[pid]
                y_train_b = y_train_batches[pid]
                train_mask_b = train_mask_batches[pid]
                loss, pred, labels = train(
                    model.train(), criterion, optimizer,
                    features_b, support_b, y_train_b, train_mask_b, torch.device('cuda'))
                avg_loss += loss
                n_nodes += pred.squeeze().numel()
                total_correct += torch.eq(pred.squeeze(), labels.squeeze()).sum().item()
        else:
            np.random.shuffle(idx_parts)
            for pid in idx_parts:
                # use preprocessed batch data
                features_b = features_batches[pid]
                support_b = support_batches[pid]
                y_train_b = y_train_batches[pid]
                train_mask_b = train_mask_batches[pid]
                loss, pred, labels = train(
                    model.train(), criterion, optimizer,
                    features_b, support_b, y_train_b, train_mask_b, torch.device('cuda'))
                avg_loss = loss.item()
                n_nodes += pred.squeeze().numel()
                total_correct += torch.eq(pred.squeeze(), labels.squeeze()).sum().item()
        train_acc = total_correct / n_nodes
        # Write Train stats to tensorboard
        writer.add_scalar('time/train', time.time() - start, epoch)
        writer.add_scalar('loss/train', avg_loss/len(features_batches), epoch)
        writer.add_scalar('acc/train', train_acc, epoch)

        # Validation
        cost, acc, micro, macro = evaluate(
            model.eval(), criterion, val_features_batches, val_support_batches, y_val_batches, val_mask_batches,
            val_nodes, torch.device("cuda"))

        # Write Valid stats to tensorboard
        writer.add_scalar('acc/valid', acc, epoch)
        writer.add_scalar('mi_F1/valid', micro, epoch)
        writer.add_scalar('ma_F1/valid', macro, epoch)
        writer.add_scalar('loss/valid', cost, epoch)
        pbar.set_postfix({"t": avg_loss/len(features_batches),"t_acc": train_acc, "v": cost, "v_acc": acc})
        pbar.update()
    pbar.close()

    # Test
    if args.test == 1:
        # Test on cpu
        cost, acc, micro, macro = test(
            model.eval(), criterion, test_features_batches, test_support_batches, y_test_batches, test_mask_batches,
            torch.device("cpu"))
        writer.add_scalar('acc/test', acc, epoch)
        writer.add_scalar('mi_F1/test', micro, epoch)
        writer.add_scalar('ma_F1/test', macro, epoch)
        writer.add_scalar('loss/test', cost, epoch)
        print('test: acc: {:.4f}'.format(acc))
        print('test: mi_f1: {:.4f}, ma_f1: {:.4f}'.format(micro, macro))


if __name__ == '__main__':
    main()
