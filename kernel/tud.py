from graph_tool.all import *

from wl import compute_wl, compute_wl_f
from auxiliarymethods.auxiliary_methods import read_txt

from auxiliarymethods.svm import linear_svm_evaluation
from auxiliarymethods.svm import normalize_feature_vector_dense
from auxiliarymethods.graphs import create_cycle, create_clique


cycles = []
for i in range(3,7):
    cycles.append(create_cycle(i))

cliques = []
for i in range(3,7):
    cliques.append(create_clique(i))

datasets = ["ENZYMES","PTC_FM", "PTC_MR", "MUTAG",  "PROTEINS", "Mutagenicity",  "MCF-7",]

for ds in datasets:
    print(ds)

    # 1-WL.
    gram_matrices = []
    for i in range(1,6):
        graph_db, classes = read_txt(ds)
        gram_matrix = compute_wl(graph_db, i, compute_gram=False)
        gram_matrix = normalize_feature_vector_dense(gram_matrix)
        gram_matrices.append(gram_matrix)

    acc, std, mrg, mrg_std = linear_svm_evaluation(gram_matrices, classes, num_repetitions=10)
    print(acc, std, mrg, mrg_std)
    print("#")
print("###")

# 1-WL_F.
for ds in datasets:
    print(ds)

    for s in range(1, (len(cycles)+1)):
        gram_matrices = []
        for i in range(1,6):
            graph_db, classes = read_txt(ds)
            gram_matrix = compute_wl_f(graph_db, cycles[0:s], i, induced=True, compute_gram=False)
            gram_matrix = normalize_feature_vector_dense(gram_matrix)
            gram_matrices.append(gram_matrix)

        acc, std, mrg, mrg_std = linear_svm_evaluation(gram_matrices, classes, num_repetitions=10)
        print(acc, std, mrg, mrg_std)
        print("#")
print("###")

for ds in datasets:
    print(ds)

    for s in range(1, (len(cliques)+1)):
        gram_matrices = []
        for i in range(1,6):
            graph_db, classes = read_txt(ds)
            gram_matrix = compute_wl_f(graph_db, cliques[0:s], i, induced=True, compute_gram=False)
            gram_matrix = normalize_feature_vector_dense(gram_matrix)
            gram_matrices.append(gram_matrix)

        acc, std, mrg, mrg_std = linear_svm_evaluation(gram_matrices, classes, num_repetitions=10)
        print(acc, std, mrg, mrg_std)
        print("#")
print("###")
