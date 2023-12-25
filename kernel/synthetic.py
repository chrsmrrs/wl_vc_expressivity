import math as m

import graph_tool as gt
import numpy as np
from graph_tool.all import *

from auxiliarymethods.graphs import create_cycle
from auxiliarymethods.svm import linear_svm_evaluation
from auxiliarymethods.svm import normalize_feature_vector_dense


# Simple implementation of 1-WL_F.
def compute_wl_f(graph_db, f_list, num_it, induced, compute_gram):
    num_subgraphs = len(f_list)
    offset = 0
    graph_indices = []
    color_manager = {}
    c = 0

    # Manage labels and indices.
    for g in graph_db:
        graph_indices.append((offset, offset + g.num_vertices() - 1))
        offset += g.num_vertices()
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
        for s, f in enumerate(f_list):
            # Compute subgraph isomorphisms from f to g.
            maps = gt.topology.subgraph_isomorphism(f, g, induced=induced)

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

    # Compute 1-WL over node-labeled graphs.
    # Compute 0th iteration.
    colors = []
    for i, g in enumerate(graph_db):
        for v in g.vertices():
            colors.append(g.vp.nl[v])

    max_all = int(np.amax(colors) + 1)
    feature_vectors = [np.bincount(colors[index[0]:index[1] + 1], minlength=max_all) for
                       i, index in enumerate(graph_indices)]

    c = 1
    while c <= num_it:
        colors = []

        for i, g in enumerate(graph_db):
            for v in g.vertices():
                neighbors = []

                for w in v.out_neighbors():
                    neighbors.append(hash(g.vp.nl[w]))

                neighbors.sort()
                neighbors.append(g.vp.nl[v])
                colors.append(hash(tuple(neighbors)))

        _, colors = np.unique(colors, return_inverse=True)

        # Assign new colors to vertices.
        q = 0
        for i, g in enumerate(graph_db):
            for v in g.vertices():
                g.vp.nl[v] = colors[q]
                q += 1

        max_all = int(np.amax(colors) + 1)

        feature_vectors = [np.bincount(colors[index[0]:index[1] + 1], minlength=max_all) for i, index in
                           enumerate(graph_indices)]
        c += 1

    feature_vectors = np.array(feature_vectors, dtype=np.double)

    # Comptue Gram matrix.
    if compute_gram:
        feature_vectors = np.dot(feature_vectors, feature_vectors.transpose())

    return feature_vectors


# Simple implementation of 1-WL.
def compute_wl(graph_db, num_it, compute_gram):
    offset = 0
    graph_indices = []

    # Manage labels and indices.
    for g in graph_db:
        graph_indices.append((offset, offset + g.num_vertices() - 1))
        offset += g.num_vertices()
        g.vp.nl = g.new_vertex_property("int")

    # Set all node labels to uniform color.
    for i, g in enumerate(graph_db):
        for v in g.vertices():
            g.vp.nl[v] = 0

    # Compute 0th iteration.
    colors = []
    for i, g in enumerate(graph_db):
        for v in g.vertices():
            colors.append(g.vp.nl[v])

    max_all = int(np.amax(colors) + 1)
    feature_vectors = [np.bincount(colors[index[0]:index[1] + 1], minlength=max_all) for
                       i, index in enumerate(graph_indices)]

    c = 1
    while c <= num_it:
        colors = []

        for i, g in enumerate(graph_db):
            for v in g.vertices():
                neighbors = []

                for w in v.out_neighbors():
                    neighbors.append(hash(g.vp.nl[w]))

                neighbors.sort()
                neighbors.append(g.vp.nl[v])
                colors.append(hash(tuple(neighbors)))

        _, colors = np.unique(colors, return_inverse=True)

        # Assign new colors to vertices.
        q = 0
        for i, g in enumerate(graph_db):
            for v in g.vertices():
                g.vp.nl[v] = colors[q]
                q += 1

        max_all = int(np.amax(colors) + 1)

        feature_vectors = [np.bincount(colors[index[0]:index[1] + 1], minlength=max_all) for i, index in
                           enumerate(graph_indices)]

        c += 1

    feature_vectors = np.array(feature_vectors, dtype=np.double)

    if compute_gram:
        feature_vectors = np.dot(feature_vectors, feature_vectors.transpose())

    return feature_vectors


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


# Sample ER graphs.
def create_er_graph(n, p):
    g = Graph(directed=False)

    for i in range(n):
        g.add_vertex()

    for i in range(n):
        for j in range(i + 1, n):
            s = np.random.uniform()

            if s <= p and not g.edge(i, j):
                g.add_edge(i, j)

    return g


# Create random graphs and classify them according to subgraph counts.
def create_random_graphs(num, n, p, t, f):
    graph_db = []
    classes = []

    for i in range(num):
        graph_db.append(create_er_graph(n, p))

    for g in graph_db:
        maps = gt.topology.subgraph_isomorphism(f, g, induced=induced)

        # if len(maps) >= t:
        #     classes.append(0)
        # else:
        #     classes.append(1)
        classes.append(len(maps))

    classes = np.array(classes)

    return graph_db, classes


# Create subgraphs.
subgraphs = []

# C_3.
g_1 = Graph(directed=False)
a = g_1.add_vertex()
b = g_1.add_vertex()
c = g_1.add_vertex()
g_1.add_edge(a, b)
g_1.add_edge(b, c)
g_1.add_edge(c, a)
subgraphs.append(g_1)

# C_4.
g_2 = Graph(directed=False)
a = g_2.add_vertex()
b = g_2.add_vertex()
c = g_2.add_vertex()
d = g_2.add_vertex()
g_2.add_edge(a, b)
g_2.add_edge(b, c)
g_2.add_edge(c, d)
g_2.add_edge(d, a)
subgraphs.append(g_2)

# C_5.
g_3 = Graph(directed=False)
a = g_3.add_vertex()
b = g_3.add_vertex()
c = g_3.add_vertex()
d = g_3.add_vertex()
e = g_3.add_vertex()
g_3.add_edge(a, b)
g_3.add_edge(b, c)
g_3.add_edge(c, d)
g_3.add_edge(d, e)
g_3.add_edge(e, a)
subgraphs.append(g_3)

# K_4.
g_4 = Graph(directed=False)
a = g_4.add_vertex()
b = g_4.add_vertex()
c = g_4.add_vertex()
d = g_4.add_vertex()
g_4.add_edge(a, b)
g_4.add_edge(b, c)
g_4.add_edge(c, d)
g_4.add_edge(d, a)
g_4.add_edge(a, c)
g_4.add_edge(b, d)
subgraphs.append(g_4)

if False:

    # First synthetic dataset, linear separability.
    num_it = 6
    for n in [16, 32, 64, 128]:
        graph_db, classes = create_linear_dataset(1000, n)
        f = create_cycle(n - 4)

        gram_matrices = []
        for i in range(num_it):
            gram_matrix = compute_wl(graph_db, i, compute_gram=False)
            gram_matrix = normalize_feature_vector_dense(gram_matrix)
            gram_matrices.append(gram_matrix)

        acc, std, mrg, mrg_std = linear_svm_evaluation(gram_matrices, classes, num_repetitions=10, C=[10 ** 7])
        print(acc, std, mrg, mrg_std)
    print("###")

    for n in [16, 32, 64, 128]:
        graph_db, classes = create_linear_dataset(1000, n)
        f = create_cycle(n - 4)

        gram_matrices = []
        for i in range(num_it):
            gram_matrix = compute_wl_f(graph_db, [f], i, induced=True, compute_gram=False)
            gram_matrix = normalize_feature_vector_dense(gram_matrix)
            gram_matrices.append(gram_matrix)

        acc, std, mrg, mrg_std = linear_svm_evaluation(gram_matrices, classes, num_iter=1000, num_repetitions=10,
                                                       C=[10 ** 10])
        print(acc, std, mrg, mrg_std)

# Some hyperparameters.
num_it = 6
induced = True
ps = [0.05, 0.1, 0.2, 0.3]
# ts = [8, 16, 32]
num_graphs = 1000
num_vertices = 20

datasets = []

for p in ps:
    for f in subgraphs:
        graph_db, classes = create_random_graphs(num_graphs, num_vertices, p, -1, f)
        datasets.append((graph_db, classes, p, f))

for (graph_db, classes, p, f) in datasets:
    print(p)
    gram_matrices = []

    if len(np.unique(classes)) >= 2:
        for i in range(num_it):
            gram_matrix = compute_wl_f(graph_db, [f], i, induced=induced, compute_gram=False)
            gram_matrix = normalize_feature_vector_dense(gram_matrix)
            gram_matrices.append(gram_matrix)

        acc, std, mrg, mrg_std = linear_svm_evaluation(gram_matrices, classes, num_iter=1000, num_repetitions=10,
                                                       C=[10 ** 10])
        print(acc, std, mrg, mrg_std)
    else:
        print("SKIP!")
    print("#")
print("###")

# 1-WL.
for (graph_db, classes, p, f) in datasets:
    print(p)
    gram_matrices = []

    if len(np.unique(classes)) >= 2:
        for i in range(num_it):
            gram_matrix = compute_wl(graph_db, i, compute_gram=False)
            gram_matrix = normalize_feature_vector_dense(gram_matrix)
            gram_matrices.append(gram_matrix)

        acc, std, mrg, mrg_std = linear_svm_evaluation(gram_matrices, classes, num_iter=1000, num_repetitions=10,
                                                       C=[10 ** 10])
        print(acc, std, mrg, mrg_std)
    else:
        print("SKIP!")
    print("#")

print("###")
