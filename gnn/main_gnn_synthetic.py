from auxiliarymethods_gnn.gnn_evaluation import gnn_evaluation_synthetic_linear_f
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


    results = []
    # for n in [16, 32, 64, 128]:
    #
    #     print(n)
    #
    #     # GIN, dataset d, layers in [1:6], hidden dimension in {32,64,128}.
    #     acc, s_1 = gnn_evaluation_synthetic_linear(GIN, n, [1, 2, 3, 4, 5], [64], max_num_epochs=200, batch_size=128,
    #                                                start_lr=0.01, num_repetitions=num_reps, all_std=False)
    #     print("GIN " + str(acc) + " " + str(s_1))
    #     results.append("GIN " + str(acc) + " " + str(s_1))

    for n in [16, 32, 64, 128]:

        print(n)

        # GIN, dataset d, layers in [1:6], hidden dimension in {32,64,128}.
        acc, s_1 = gnn_evaluation_synthetic_linear_f(GIN, n, [1,2,3,4,5], [64], max_num_epochs=200, batch_size=128,
                                       start_lr=0.01, num_repetitions=num_reps, all_std=False)
        print("GIN " + str(acc) + " " + str(s_1))
        results.append("GIN " + str(acc) + " " + str(s_1))

    # Create subgraphs.
    subgraphs = []






if __name__ == "__main__":
    main()
