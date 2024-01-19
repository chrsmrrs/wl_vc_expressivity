import math as m

import graph_tool as gt
import numpy as np
from graph_tool.all import *

from auxiliarymethods.graphs import create_cycle
from auxiliarymethods.svm import kernel_svm_evaluation
from auxiliarymethods.svm import normalize_gram_matrix

from wl import compute_wloa, compute_wloa_f
from auxiliarymethods.svm import linear_svm_evaluation
from auxiliarymethods.svm import normalize_feature_vector_dense

from wl import compute_wl, compute_wl_f


# Create dataset not linear separable by 1-WL.
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
                g.add_vertex()le

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


# # First synthetic dataset, linear separability.
# num_it = 6
# for n in [16, 32, 64, 128]:
#     graph_db, classes = create_linear_dataset(1000, n)
#     f = create_cycle(n - 4)
#
#     gram_matrices = []
#     for i in range(num_it):
#         gram_matrix = compute_wloa(graph_db, i)
#         gram_matrix = normalize_gram_matrix(gram_matrix)
#         gram_matrices.append(gram_matrix)
#
#     acc, std = kernel_svm_evaluation(gram_matrices, classes, num_repetitions=10, C=[10 ** 10])
#     print(acc, std)
# print("###")
#
# for n in [16, 32, 64, 128]:
#     graph_db, classes = create_linear_dataset(1000, n)
#     f = create_cycle(n - 4)
#
#     gram_matrices = []
#     for i in range(num_it):
#         gram_matrix = compute_wloa_f(graph_db, [f], i, induced=True)
#         gram_matrix = normalize_gram_matrix(gram_matrix)
#         gram_matrices.append(gram_matrix)
#
#     acc, std= kernel_svm_evaluation(gram_matrices, classes, num_repetitions=10,
#                                                    C=[10 ** 10])
#     print(acc, std)
#

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

# 1-WLOA.
results = []
for (graph_db, classes, p, f) in datasets:
    print(p)
    gram_matrices = []

    if len(np.unique(classes)) >= 2:
        for i in range(num_it):
            gram_matrix = compute_wloa(graph_db, i)
            gram_matrix = normalize_gram_matrix(gram_matrix)
            gram_matrices.append(gram_matrix)

        acc_train, std_train, acc_test, std_test = kernel_svm_evaluation(gram_matrices, classes, num_repetitions=10,
                                                       C=[10 ** 10])
        print(acc_train, std_train, acc_test, std_test)
        results.append([p,acc_train, std_train, acc_test, std_test])
    else:
        print("SKIP!")
    print("#")
print("###")


# 1-WLOA_F.
for (graph_db, classes, p, f) in datasets:
    print(p)
    gram_matrices = []

    if len(np.unique(classes)) >= 2:
        for i in range(num_it):
            gram_matrix = compute_wloa_f(graph_db, [f], i, induced=induced)
            gram_matrix = normalize_gram_matrix(gram_matrix)
            gram_matrices.append(gram_matrix)

        acc_train, std_train, acc_test, std_test = kernel_svm_evaluation(gram_matrices, classes, num_repetitions=10,
                                                       C=[10 ** 10])
        print(acc_train, std_train, acc_test, std_test)
        results.append([p,acc_train, std_train, acc_test, std_test])
    else:
        print("SKIP!")
    print("#")
print("###")


# 1-WL_F.
for (graph_db, classes, p, f) in datasets:
    print(p)
    gram_matrices = []

    if len(np.unique(classes)) >= 2:
        for i in range(num_it):
            gram_matrix = compute_wl_f(graph_db, [f], i, induced=induced, compute_gram=False)
            gram_matrix = normalize_feature_vector_dense(gram_matrix)
            gram_matrices.append(gram_matrix)

        acc_train, std_train, acc_test, std_test, mrg, mrg_std = linear_svm_evaluation(gram_matrices, classes, num_iter=1000, num_repetitions=10,
                                                        C=[10 ** 10], all=False)

        print(p, acc_train, std_train, acc_test, std_test, mrg, mrg_std)

        results.append([p, acc_train, std_train, acc_test, std_test, mrg, mrg_std])

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

        acc_train, std_train, acc_test, std_test, mrg, mrg_std = linear_svm_evaluation(gram_matrices, classes, num_iter=1000, num_repetitions=10,
                                                       C=[10 ** 10], all=False)

        print(p, acc_train, std_train, acc_test, std_test, mrg, mrg_std)

        results.append([p, acc_train, std_train, acc_test, std_test, mrg, mrg_std])
    else:
        print("SKIP!")
    print("#")
print("###")

for r in results:
    print(r)



