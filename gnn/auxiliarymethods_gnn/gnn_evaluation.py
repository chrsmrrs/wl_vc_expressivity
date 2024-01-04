import math as m
import os.path as osp

import graph_tool as gt
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from kernel.auxiliarymethods.auxiliary_methods import read_txt
from graph_tool.all import *
from graph_tool.all import *
from graph_tool.all import *
from graph_tool.all import *
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from torch_geometric.data import DataLoader
from torch_geometric.data import (InMemoryDataset, Data)
from torch_geometric.utils import degree


class wl(InMemoryDataset):
    def __init__(self, root, dataset, transform=None, pre_transform=None,
                 pre_filter=None):
        self.dataset = dataset

        super(wl, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return self.dataset

    @property
    def processed_file_names(self):
        return self.dataset

    def download(self):
        pass

    def process(self):
        graph_db, classes = read_txt(self.dataset)

        # Normalize class labels.
        _, classes_new = np.unique(classes, return_inverse=True)
        classes = list(classes_new)

        for g in graph_db:
            g.vp.nl = g.new_vertex_property("int")

        for g in graph_db:
            for v in g.vertices():
                g.vp.nl[v] = 0

        matrices = []
        labels = []
        for g in graph_db:
            a = []
            b = []
            x = []
            for (i, j) in g.edges():
                a.append(int(i))
                b.append(int(j))

                # Other direction.
                a.append(int(j))
                b.append(int(i))

            for v in g.vertices():
                x.append(g.vp.nl[v])

            edge_index = torch.tensor([a, b])
            x = np.array(x)

            matrices.append(edge_index)
            labels.append(x)

        data_list = []
        for i, m in enumerate(matrices):
            data = Data()
            data.edge_index = m

            one_hot = np.eye(64)[labels[i]]
            data.x = torch.from_numpy(one_hot).to(torch.float)

            data.y = torch.from_numpy(np.array(classes[i])).to(torch.long)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class MyData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        return self.num_nodes if key in [
            'edge_index'
        ] else 0


class MyTransform(object):
    def __call__(self, data):
        new_data = MyData()
        for key, item in data:
            new_data[key] = item
        return new_data


# Create cycle on n vertices.
def create_cycle(n):
    g = Graph(directed=False)

    for i in range(n):
        g.add_vertex()

    for i in range(n - 1):
        g.add_edge(i, i + 1)

    g.add_edge(n - 1, 0)

    return g


# Create dataset not linear separabel by 1-WL.
def create_linear_dataset(num, n):
    classes = []
    graph_db = []

    for i in range(1, num + 1):
        # Even.
        if i % 2 == 0:
            g = Graph(directed=False)

            for _ in range(i):
                g.add_vertex()

            c = create_cycle(n - 4)
            g = graph_union(g, c)

            graph_db.append(g)
            classes.append(0)
        # Odd.
        else:
            g = Graph(directed=False)

            for _ in range(i):
                g.add_vertex()

            c_1 = create_cycle(m.ceil(n / 2) - 2)
            c_2 = create_cycle(m.ceil(n / 2) - 2)

            g = graph_union(g, c_1)
            g = graph_union(g, c_2)

            graph_db.append(g)
            classes.append(1)

    return graph_db, np.array(classes)


class synthetic_linear(InMemoryDataset):
    def __init__(self, root, n, transform=None, pre_transform=None,
                 pre_filter=None):
        self.n = n
        super(synthetic_linear, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return str(self.n)

    @property
    def processed_file_names(self):
        return str(self.n)

    def download(self):
        pass

    def process(self):

        color_manager = {}
        c = 0

        graph_db, classes = create_linear_dataset(1000, self.n)

        for g in graph_db:
            g.vp.nl = g.new_vertex_property("int")
            g.vp.labels = g.new_vertex_property("vector<int>")

        # Set all node labels to uniform color.
        for i, g in enumerate(graph_db):
            for v in g.vertices():
                g.vp.nl[v] = 0

        matrices = []
        labels = []
        for g in graph_db:
            a = []
            b = []
            x = []
            for (i, j) in g.edges():
                a.append(int(i))
                b.append(int(j))

            for v in g.vertices():
                x.append(g.vp.nl[v])

            edge_index = torch.tensor([a, b])
            x = np.array(x)

            matrices.append(edge_index)
            labels.append(x)

        data_list = []
        for i, m in enumerate(matrices):
            data = Data()
            data.edge_index = m

            one_hot = np.eye(64)[labels[i]]
            data.x = torch.from_numpy(one_hot).to(torch.float)

            data.y = torch.from_numpy(np.array(classes[i])).to(torch.long)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class wl_f_linear_synthetic(InMemoryDataset):
    def __init__(self, root, n, subgraphs, transform=None, pre_transform=None,
                 pre_filter=None):
        self.subgraphs = subgraphs
        self.n = n

        super(wl_f_linear_synthetic, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return str(self.n) + "F"

    @property
    def processed_file_names(self):
        return str(self.n) + "F"

    def download(self):
        pass

    def process(self):
        num_subgraphs = len(self.subgraphs)
        color_manager = {}
        c = 0
        graph_db, classes = create_linear_dataset(1000, self.n)

        for g in graph_db:
            g.vp.nl = g.new_vertex_property("int")
            g.vp.labels = g.new_vertex_property("vector<int>")

        # Set all node labels to uniform color.
        for i, g in enumerate(graph_db):
            for v in g.vertices():
                g.vp.nl[v] = 0
                g.vp.labels[v] = [0] * num_subgraphs

        # Label node according to subgraphs.
        for i, g in enumerate(graph_db):
            # Iterate over subgraphs.
            for s, f in enumerate(self.subgraphs):
                # Compute subgraph isomorphisms from f to g.
                maps = gt.topology.subgraph_isomorphism(f, g, induced=True)

                for m in maps:
                    for v in f.vertices():
                        g.vp.labels[m[v]][s] += 1

        # Compress vector labels.
        for i, g in enumerate(graph_db):
            for v in g.vertices():
                h = hash(tuple(g.vp.labels[v]))

                if h in color_manager:
                    g.vp.nl[v] = color_manager[h]
                else:
                    color_manager[h] = c
                    g.vp.nl[v] = c
                    c += 1

        matrices = []
        labels = []
        for g in graph_db:
            a = []
            b = []
            x = []
            for (i, j) in g.edges():
                a.append(int(i))
                b.append(int(j))

            for v in g.vertices():
                x.append(g.vp.nl[v])

            edge_index = torch.tensor([a, b])
            x = np.array(x)

            matrices.append(edge_index)
            labels.append(x)

        data_list = []
        for i, m in enumerate(matrices):
            data = Data()
            data.edge_index = m

            one_hot = np.eye(64)[labels[i]]
            data.x = torch.from_numpy(one_hot).to(torch.float)

            data.y = torch.from_numpy(np.array(classes[i])).to(torch.long)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class MyData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        return self.num_nodes if key in [
            'edge_index'
        ] else 0


class MyTransform(object):
    def __call__(self, data):
        new_data = MyData()
        for key, item in data:
            new_data[key] = item
        return new_data


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


# One training epoch for GNN model.
def train(train_loader, model, optimizer, device):
    model.train()

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        optimizer.step()


# Get acc. of GNN model.
def test(loader, model, device):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


# 10-CV for GNN training and hyperparameter selection.
def gnn_evaluation(gnn, ds_name, layers, hidden, max_num_epochs=200, batch_size=128, start_lr=0.01, min_lr=0.000001,
                   factor=0.5, patience=5,
                   num_repetitions=10, all_std=True):
    # Load dataset and shuffle.
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', ds_name)
    dataset = wl(path, ds_name, transform=MyTransform()).shuffle()

    # Set device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_accuracies_all = []
    test_accuracies_complete = []

    for i in range(num_repetitions):
        # Test acc. over all folds.
        test_accuracies = []
        kf = KFold(n_splits=10, shuffle=True)
        dataset.shuffle()

        for train_index, test_index in kf.split(list(range(len(dataset)))):
            # Sample 10% split from training split for validation.
            train_index, val_index = train_test_split(train_index, test_size=0.1)
            best_val_acc = 0.0
            best_test = 0.0

            # Split data.
            train_dataset = dataset[train_index.tolist()]
            val_dataset = dataset[val_index.tolist()]
            test_dataset = dataset[test_index.tolist()]

            # Prepare batching.
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

            # Collect val. and test acc. over all hyperparameter combinations.
            for l in layers:
                for h in hidden:
                    # Setup model.
                    model = gnn(dataset, l, h).to(device)
                    model.reset_parameters()

                    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                           factor=factor, patience=patience,
                                                                           min_lr=0.0000001)
                    for epoch in range(1, max_num_epochs + 1):
                        lr = scheduler.optimizer.param_groups[0]['lr']
                        train(train_loader, model, optimizer, device)
                        val_acc = test(val_loader, model, device)
                        scheduler.step(val_acc)

                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            best_test = test(test_loader, model, device) * 100.0

                        # Break if learning rate is smaller 10**-6.
                        if lr < min_lr:
                            break

            test_accuracies.append(best_test)

            if all_std:
                test_accuracies_complete.append(best_test)
        test_accuracies_all.append(float(np.array(test_accuracies).mean()))

    if all_std:
        return (np.array(test_accuracies_all).mean(), np.array(test_accuracies_all).std(),
                np.array(test_accuracies_complete).std())
    else:
        return (np.array(test_accuracies_all).mean(), np.array(test_accuracies_all).std())


# 10-CV for GNN training and hyperparameter selection.
def gnn_evaluation_synthetic_linear_f(gnn, num_nodes, layers, hidden, max_num_epochs=200, batch_size=128, start_lr=0.01,
                                      min_lr=0.000001, factor=0.5, patience=5,
                                      num_repetitions=10, all_std=True):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', "test")
    dataset = wl_f_linear_synthetic(path, num_nodes, [create_cycle(num_nodes - 4)], transform=MyTransform()).shuffle()

    # Set device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_accuracies_all = []
    test_accuracies_complete = []

    for i in range(num_repetitions):
        # Test acc. over all folds.
        test_accuracies = []
        kf = KFold(n_splits=10, shuffle=True)
        dataset.shuffle()

        for train_index, test_index in kf.split(list(range(len(dataset)))):
            # Sample 10% split from training split for validation.
            train_index, val_index = train_test_split(train_index, test_size=0.1)
            best_val_acc = 0.0
            best_test = 0.0

            # Split data.
            train_dataset = dataset[train_index.tolist()]
            val_dataset = dataset[val_index.tolist()]
            test_dataset = dataset[test_index.tolist()]

            # Prepare batching.
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

            # Collect val. and test acc. over all hyperparameter combinations.
            for l in layers:
                for h in hidden:
                    # Setup model.
                    model = gnn(dataset, l, h).to(device)
                    model.reset_parameters()

                    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                           factor=factor, patience=patience,
                                                                           min_lr=0.0000001)
                    for epoch in range(1, max_num_epochs + 1):
                        lr = scheduler.optimizer.param_groups[0]['lr']
                        train(train_loader, model, optimizer, device)
                        val_acc = test(val_loader, model, device)
                        scheduler.step(val_acc)

                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            best_test = test(test_loader, model, device) * 100.0

                        # Break if learning rate is smaller 10**-6.
                        if lr < min_lr:
                            break

            test_accuracies.append(best_test)

            if all_std:
                test_accuracies_complete.append(best_test)
        test_accuracies_all.append(float(np.array(test_accuracies).mean()))

    if all_std:
        return (np.array(test_accuracies_all).mean(), np.array(test_accuracies_all).std(),
                np.array(test_accuracies_complete).std())
    else:
        return (np.array(test_accuracies_all).mean(), np.array(test_accuracies_all).std())
