from auxiliarymethods_gnn.gnn_evaluation import gnn_evaluation_synthetic_linear, gnn_evaluation_synthetic_linear_f
from gnn_baselines.gnn_architectures import GIN, GINE, GINEWithJK, GINWithJK


def main():
    num_reps = 10


    results = []
    for n in [16, 32, 64, 128]:

        print(n)

        # GIN, dataset d, layers in [1:6], hidden dimension in {32,64,128}.
        acc, s_1 = gnn_evaluation_synthetic_linear(GIN, n, [1,2,3,4,5], [64], max_num_epochs=200, batch_size=128,
                                       start_lr=0.01, num_repetitions=num_reps, all_std=False)
        print("GIN " + str(acc) + " " + str(s_1))
        results.append("GIN " + str(acc) + " " + str(s_1))

    for n in [16, 32, 64, 128]:

        print(n)

        # GIN, dataset d, layers in [1:6], hidden dimension in {32,64,128}.
        acc, s_1 = gnn_evaluation_synthetic_linear_f(GIN, n, [1,2,3,4,5], [64], max_num_epochs=200, batch_size=128,
                                       start_lr=0.01, num_repetitions=num_reps, all_std=False)
        print("GIN " + str(acc) + " " + str(s_1))
        results.append("GIN " + str(acc) + " " + str(s_1))


if __name__ == "__main__":
    main()
