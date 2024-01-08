from auxiliarymethods_gnn.gnn_evaluation import gnn_evaluation_synthetic_er, gnn_evaluation_synthetic_er_f
from gnn_baselines.gnn_architectures import GIN, GINE, GINEWithJK, GINWithJK

import graph_tool as gt
import numpy as np
from graph_tool.all import *

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


def main():
    num_reps = 10

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

    num_it = 6
    induced = True
    ps = [0.05, 0.1, 0.2, 0.3]
    num_graphs = 1000
    num_vertices = 20

    datasets = []

    for p in ps:
        for f in subgraphs:
            acc, s_1 =gnn_evaluation_synthetic_er(GIN, p, num_graphs, num_vertices, f, [1, 2, 3, 4, 5], [64], max_num_epochs=200, batch_size=128,
                                                   start_lr=0.01, num_repetitions=num_reps, all_std=False)


            print("GIN " + str(acc) + " " + str(s_1))

    for p in ps:
        for f in subgraphs:
            acc, s_1 =gnn_evaluation_synthetic_er_f(GIN, p, num_graphs, num_vertices, f, [1, 2, 3, 4, 5], [64], max_num_epochs=200, batch_size=128,
                                                   start_lr=0.01, num_repetitions=num_reps, all_std=False)


            print("GIN " + str(acc) + " " + str(s_1))







if __name__ == "__main__":
    main()
