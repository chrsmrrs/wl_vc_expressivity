import matplotlib
#matplotlib.use("TKAgg")
#matplotlib.use("macOSX")
from matplotlib import pyplot as plt

import math as m

import graph_tool as gt
import numpy as np
from graph_tool.all import *

from auxiliarymethods.graphs import create_cycle
from auxiliarymethods.svm import linear_svm_evaluation
from auxiliarymethods.svm import normalize_feature_vector_dense

from wl import compute_wl, compute_wl_f



import seaborn as sns
import pandas as pd

import numpy as np

# for using latex in plt it requires one installation:
# $ sudo apt install dvipng
#rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
#rc('text', usetex=True)

sns.set_theme(style="white")



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


#
# # First synthetic dataset, linear separability.
# num_it = 6
# for n in [16, 32, 64, 128]:
#     graph_db, classes = create_linear_dataset(1000, n)
#     f = create_cycle(n - 4)
#
#     gram_matrices = []
#     for i in range(num_it):
#         gram_matrix = compute_wl(graph_db, i, compute_gram=False)
#         gram_matrix = normalize_feature_vector_dense(gram_matrix)
#         gram_matrices.append(gram_matrix)
#
#     acc, std, mrg, mrg_std = linear_svm_evaluation(gram_matrices, classes, num_repetitions=10, C=[10 ** 10])
#     print(acc, std, mrg, mrg_std)
# print("###")
#
# for n in [16, 32, 64, 128]:
#     graph_db, classes = create_linear_dataset(1000, n)
#     f = create_cycle(n - 4)
#
#     gram_matrices = []
#     for i in range(num_it):
#         gram_matrix = compute_wl_f(graph_db, [f], i, induced=True, compute_gram=False)
#         gram_matrix = normalize_feature_vector_dense(gram_matrix)
#         gram_matrices.append(gram_matrix)
#
#     acc, std, mrg, mrg_std = linear_svm_evaluation(gram_matrices, classes, num_iter=1000, num_repetitions=10,
#                                                    C=[10 ** 10])
#     print(acc, std, mrg, mrg_std)


# Some hyperparameters.
num_it = 6
induced = True
ps = [0.050, 0.075, 0.100, 0.125, 0.150, 0.175, 0.200, 0.225, 0.250, 0.275, 0.30, 0.325, 0.350, 0.375, 0.400]
#ps = [0.050, 0.075]
# ts = [8, 16, 32]
num_graphs = 1000
num_vertices = 20

datasets = []
subgraphs = [g_2]

for p in ps:
    for f in subgraphs:
        graph_db, classes = create_random_graphs(num_graphs, num_vertices, p, -1, f)
        datasets.append((graph_db, classes, p, f))


data = np.zeros([0,3])

results = []
for (graph_db, classes, p, f) in datasets:
    print(p)
    gram_matrices = []

    if len(np.unique(classes)) >= 2:
        for i in range(num_it):
            gram_matrix = compute_wl_f(graph_db, [f], i, induced=induced, compute_gram=False)
            gram_matrix = normalize_feature_vector_dense(gram_matrix)
            gram_matrices.append(gram_matrix)

        # acc_train, std_train, acc_test, std_test, mrg, mrg_std = linear_svm_evaluation(gram_matrices, classes, num_iter=1000, num_repetitions=10,
        #                                                C=[10 ** 10])

        acc_train, acc_test, mrg = linear_svm_evaluation(gram_matrices, classes, num_iter=1000, num_repetitions=10,
                                                       C=[10 ** 10])

        mrg = mrg.mean()

        acc_train = np.reshape(acc_train, [10, 1])
        acc_test = np.reshape(acc_test, [10, 1])
        mrg = np.reshape(np.array([mrg]*10), [10, 1])
        ps = np.reshape(np.array([p]*10), [10,1])

        acc_diff = acc_train - acc_test

        matrix = np.concatenate([acc_diff, mrg, ps], axis=1)

        data = np.concatenate([data,matrix], axis=0)

        print(data)


    else:
        print("SKIP!")
    print("#")
print("###")

df = pd.DataFrame(data, columns=["Difference", "Margin", "p"])

g = sns.lineplot(x="Margin", y="Difference", hue="p", data=df)
g.set(xlabel="Margin $\lambda$", ylabel="Train - test accuracy", title="1-$\mathsf{WL}_\mathcal{F}$")

sns.move_legend(g, "upper left", title='Prob.')

plt.savefig('line_plot.pdf')

#plt.show()



exit()

# 1-WL.
for (graph_db, classes, p, f) in datasets:
    print(p)
    gram_matrices = []

    if len(np.unique(classes)) >= 2:
        for i in range(num_it):
            gram_matrix = compute_wl(graph_db, i, compute_gram=False)
            gram_matrix = normalize_feature_vector_dense(gram_matrix)
            gram_matrices.append(gram_matrix)

        acc_train, acc_test, mrg = linear_svm_evaluation(gram_matrices, classes, num_iter=1000, num_repetitions=10,
                                                       C=[10 ** 10])

        print(acc_train)
        print(acc_test)
        print(mrg)
        results.append((p, acc_train, acc_test, mrg))
    else:
        print("SKIP!")
    print("#")

print("###")

for r in results:
    print(r)


