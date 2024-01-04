from graph_tool.all import *

from auxiliarymethods_gnn.gnn_f_evaluation import gnn_evaluation
from auxiliarymethods_gnn.graphs import create_cycle, create_clique
from gnn_baselines.gnn_architectures import GIN

cycles = []
for i in range(3, 7):
    cycles.append(create_cycle(i))

cliques = []
for i in range(3, 7):
    cliques.append(create_clique(i))

def main():
    num_reps = 10

    datasets = ["ENZYMES", "MUTAG", "PROTEINS", "PTC_FM", "PTC_MR", "NCI1"]
    for d in datasets:
        print(d)
        for s in range(1, (len(cycles) + 1)):
            acc, s_1 = gnn_evaluation(GIN, cycles[0:s], d, [1,2,3,4,5], [64], max_num_epochs=200, batch_size=128,
                                           start_lr=0.01, num_repetitions=num_reps, all_std=False)
            print(d + " " + "GIN " + str(acc) + " " + str(s_1))

    for d in datasets:
        print(d)
        for s in range(1, (len(cliques) + 1)):
            acc, s_1 = gnn_evaluation(GIN, cliques[0:s], d, [1,2,3,4,5], [64], max_num_epochs=200, batch_size=128,
                                           start_lr=0.01, num_repetitions=num_reps, all_std=False)
            print(d + " " + "GIN " + str(acc) + " " + str(s_1))

if __name__ == "__main__":
    main()
