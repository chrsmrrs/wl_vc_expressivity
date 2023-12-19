import auxiliarymethods.datasets as dp
from auxiliarymethods.gnn_f_evaluation import gnn_evaluation
from gnn_baselines.gnn_architectures import GIN, GINE, GINEWithJK, GINWithJK

from graph_tool.all import *
import graph_tool as gt

def main():
    num_reps = 5

    ### Smaller datasets.
    dataset = [["ENZYMES", False]]

    f = Graph(directed=False)
    a = f.add_vertex()
    b = f.add_vertex()
    c = f.add_vertex()
    # d = f.add_vertex()
    f.add_edge(a, b)
    f.add_edge(b, c)
    f.add_edge(c, a)
    # f.add_edge(d, a)

    results = []
    for d, use_labels in dataset:
        # Download dataset.
        #dp.get_dataset_f(d)

        # GIN, dataset d, layers in [1:6], hidden dimension in {32,64,128}.
        acc, s_1, s_2 = gnn_evaluation(GIN, f,  d, [3], [64], max_num_epochs=200, batch_size=64,
                                       start_lr=0.01, num_repetitions=num_reps, all_std=True)
        print(d + " " + "GIN " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(d + " " + "GIN " + str(acc) + " " + str(s_1) + " " + str(s_2))





if __name__ == "__main__":
    main()
