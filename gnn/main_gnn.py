from graph_tool.all import *

from auxiliarymethods_gnn.gnn_evaluation import gnn_evaluation
from gnn_baselines.gnn_architectures import GIN

def main():
    num_reps = 10

    datasets = ["ENZYMES", "MUTAG", "PTC_FM", "PTC_MR"]
    datasets = ["PROTEINS", "NCI1"]
    for d in datasets:
        print(d)
        acc_train, s_train, acc_test, s_test = gnn_evaluation(GIN, d, [1,2,3,4,5], [64], max_num_epochs=200, batch_size=128,
                                       start_lr=0.01, num_repetitions=num_reps, all_std=False)

        print(acc_train, s_train, acc_test, s_test, acc_train - acc_test)

if __name__ == "__main__":
    main()
