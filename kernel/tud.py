from graph_tool.all import *

from auxiliarymethods.auxiliary_methods import read_txt
from auxiliarymethods.graphs import create_cycle, create_clique
from auxiliarymethods.svm import linear_svm_evaluation, kernel_svm_evaluation
from auxiliarymethods.svm import normalize_feature_vector_dense, normalize_gram_matrix
from wl import compute_wl, compute_wl_f, compute_wloa, compute_wloa_f

cycles = []
for i in range(3, 7):
    cycles.append(create_cycle(i))

cliques = []
for i in range(3, 7):
    cliques.append(create_clique(i))

datasets = ["ENZYMES", "MUTAG", "PROTEINS", "PTC_FM", "PTC_MR", "NCI1"]  # , "Mutagenicity",  "MCF-7",]
#datasets = ["ENZYMES"] #, "Mutagenicity",  "MCF-7",]
#datasets = ["PTC_MR"]  # , "Mutagenicity",  "MCF-7",]

# 1-WL.
for ds in datasets:
    print(ds)

    gram_matrices = []
    for i in range(1, 6):
        graph_db, classes = read_txt(ds)
        gram_matrix = compute_wl(graph_db, i, compute_gram=False)
        gram_matrix = normalize_feature_vector_dense(gram_matrix)
        gram_matrices.append(gram_matrix)

    acc, std, mrg, mrg_std = linear_svm_evaluation(gram_matrices, classes, num_iter=1000,
                                                   num_repetitions=10)  # , C=[10 ** 10])
    print(acc, std, mrg, mrg_std)
    print("#")
print("###")

# 1-WL_F.
for ds in datasets:
    print(ds)

    for s in range(1, (len(cycles) + 1)):
        gram_matrices = []
        for i in range(1, 6):
            graph_db, classes = read_txt(ds)
            gram_matrix = compute_wl_f(graph_db, cycles[0:s], i, induced=True, compute_gram=False)
            gram_matrix = normalize_feature_vector_dense(gram_matrix)
            gram_matrices.append(gram_matrix)

        acc, std, mrg, mrg_std = linear_svm_evaluation(gram_matrices, classes, num_iter=1000,
                                                       num_repetitions=10)  # , C=[10 ** 10])
        print(acc, std, mrg, mrg_std)
        print("#")
print("###")

for ds in datasets:
    print(ds)

    for s in range(1, (len(cycles) + 1)):
        gram_matrices = []
        for i in range(1, 6):
            graph_db, classes = read_txt(ds)
            gram_matrix = compute_wl_f(graph_db, cliques[0:s], i, induced=True, compute_gram=False)
            gram_matrix = normalize_feature_vector_dense(gram_matrix)
            gram_matrices.append(gram_matrix)

        acc, std, mrg, mrg_std = linear_svm_evaluation(gram_matrices, classes, num_iter=1000,
                                                       num_repetitions=10)  # , C=[10 ** 10])
        print(acc, std, mrg, mrg_std)
        print("#")
print("###")

exit()

# 1-WLOA.
for ds in datasets:
    print(ds)

    gram_matrices = []
    for i in range(1, 6):
        print(i)
        graph_db, classes = read_txt(ds)
        gram_matrix = compute_wloa(graph_db, i)
        gram_matrix = normalize_gram_matrix(gram_matrix)
        gram_matrices.append(gram_matrix)

    acc, std = kernel_svm_evaluation(gram_matrices, classes, num_repetitions=10)  # , C=[10 ** 10])
    print(acc, std)
    print("#")
print("###")

# 1-WLOA_F.
for ds in datasets:
    print(ds)

    for s in range(1, (len(cycles) + 1)):
        gram_matrices = []
        for i in range(1, 6):
            graph_db, classes = read_txt(ds)
            gram_matrix = compute_wloa_f(graph_db, cycles[0:s], i, induced=True)
            gram_matrix = normalize_gram_matrix(gram_matrix)
            gram_matrices.append(gram_matrix)

        acc, std = kernel_svm_evaluation(gram_matrices, classes, num_repetitions=10)  # , C=[10 ** 10])
        print(acc, std)
        print("#")
print("###")

for ds in datasets:
    print(ds)

    for s in range(1, (len(cycles) + 1)):
        gram_matrices = []
        for i in range(1, 6):
            graph_db, classes = read_txt(ds)
            gram_matrix = compute_wloa_f(graph_db, cliques[0:s], i, induced=True)
            gram_matrix = normalize_gram_matrix(gram_matrix)
            gram_matrices.append(gram_matrix)

        acc, std = kernel_svm_evaluation(gram_matrices, classes, num_repetitions=10)  # , C=[10 ** 10])
        print(acc, std)
        print("#")
print("###")
