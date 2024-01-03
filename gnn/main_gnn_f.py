from graph_tool.all import *

from auxiliarymethods_gnn.gnn_f_evaluation import gnn_evaluation
from auxiliarymethods_gnn.graphs import create_cycle
from gnn_baselines.gnn_architectures import GIN

subgraphs = []
for i in range(3, 16):
    subgraphs.append(create_cycle(i))


def main():
    num_reps = 5

    ### Smaller datasets.
    dataset = [["PTC_FM", False]]

    results = []
    for d, use_labels in dataset:

        # GIN, dataset d, layers in [1:6], hidden dimension in {32,64,128}.
        acc, s_1, s_2 = gnn_evaluation(GIN, subgraphs, d, [3], [64], max_num_epochs=200, batch_size=64,
                                       start_lr=0.01, num_repetitions=num_reps, all_std=True)
        print(d + " " + "GIN " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(d + " " + "GIN " + str(acc) + " " + str(s_1) + " " + str(s_2))


if __name__ == "__main__":
    main()
