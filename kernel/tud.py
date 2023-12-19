import graph_tool as gt
import numpy as np
from graph_tool.all import *
import math as m
from auxiliarymethods.auxiliary_methods import read_txt

from auxiliarymethods.svm import linear_svm_evaluation
from auxiliarymethods.svm import normalize_feature_vector_dense

# Simple implementation of 1-WL_F.
def compute_wl_f_all(graph_db, f_list, num_it, induced, compute_gram):
    num_subgraphs = len(f_list)
    offset = 0
    graph_indices = []
    color_manager = {}
    c = 0

    for g in graph_db:
        graph_indices.append((offset, offset + g.num_vertices() - 1))
        offset += g.num_vertices()
        g.vp.nl = g.new_vertex_property("int")

        g.vp.labels = g.new_vertex_property("vector<int>")

    # Set all node labels to uniform color.
    for i, g in enumerate(graph_db):
        for v in g.vertices():
            g.vp.nl[v] = 0
            g.vp.labels[v] = [0] * num_subgraphs

    # Label node according to f.
    for i, g in enumerate(graph_db):

        # Iterator over subgraph
        for s, f in enumerate(f_list):
            # Compute subgraph isomorphisms from f to g.
            maps = gt.topology.subgraph_isomorphism(f, g, induced=induced)

            for m in maps:
                for v in f.vertices():
                    g.vp.labels[m[v]][s] += 1

    for i, g in enumerate(graph_db):
        for v in g.vertices():
            h = hash(tuple(g.vp.labels[v]))

            if h in color_manager:
                g.vp.nl[v] = color_manager[h]
            else:
                color_manager[h] = c
                g.vp.nl[v] = c
                c += 1

    # Compute 0th iteration.
    colors = []
    for i, g in enumerate(graph_db):
        for v in g.vertices():
            colors.append(g.vp.nl[v])

    max_all = int(np.amax(colors) + 1)
    feature_vectors = [np.bincount(colors[index[0]:index[1] + 1], minlength=max_all) for
                       i, index in enumerate(graph_indices)]

    c = 1
    while c <= num_it:
        colors = []

        for i, g in enumerate(graph_db):
            for v in g.vertices():
                neighbors = []

                for w in v.out_neighbors():
                    neighbors.append(hash(g.vp.nl[w]))

                neighbors.sort()
                neighbors.append(g.vp.nl[v])
                colors.append(hash(tuple(neighbors)))

        _, colors = np.unique(colors, return_inverse=True)

        # Assign new colors to vertices.
        q = 0
        for i, g in enumerate(graph_db):
            for v in g.vertices():
                g.vp.nl[v] = colors[q]
                q += 1

        max_all = int(np.amax(colors) + 1)

        feature_vectors = [np.bincount(colors[index[0]:index[1] + 1], minlength=max_all) for i, index in
                           enumerate(graph_indices)]
        c += 1

    feature_vectors = np.array(feature_vectors, dtype=np.double)

    if compute_gram:
        feature_vectors = np.dot(feature_vectors, feature_vectors.transpose())

    return feature_vectors


# Simple implementation of 1-WL.
def compute_wl(graph_db, num_it, compute_gram):
    offset = 0
    graph_indices = []

    for g in graph_db:
        graph_indices.append((offset, offset + g.num_vertices() - 1))
        offset += g.num_vertices()
        g.vp.nl = g.new_vertex_property("int")

    # Set all node labels to uniform color.
    for i, g in enumerate(graph_db):
        for v in g.vertices():
            g.vp.nl[v] = 0

    # Compute 0th iteration.
    colors = []
    for i, g in enumerate(graph_db):
        for v in g.vertices():
            colors.append(g.vp.nl[v])

    max_all = int(np.amax(colors) + 1)
    feature_vectors = [np.bincount(colors[index[0]:index[1] + 1], minlength=max_all) for
                       i, index in enumerate(graph_indices)]

    c = 1
    while c <= num_it:
        colors = []

        for i, g in enumerate(graph_db):
            for v in g.vertices():
                neighbors = []

                for w in v.out_neighbors():
                    neighbors.append(hash(g.vp.nl[w]))

                neighbors.sort()
                neighbors.append(g.vp.nl[v])
                colors.append(hash(tuple(neighbors)))

        _, colors = np.unique(colors, return_inverse=True)

        # Assign new colors to vertices.
        q = 0
        for i, g in enumerate(graph_db):
            for v in g.vertices():
                g.vp.nl[v] = colors[q]
                q += 1

        max_all = int(np.amax(colors) + 1)

        feature_vectors = [np.bincount(colors[index[0]:index[1] + 1], minlength=max_all) for i, index in
                           enumerate(graph_indices)]

        c += 1

    feature_vectors = np.array(feature_vectors, dtype=np.double)

    if compute_gram:
        feature_vectors = np.dot(feature_vectors, feature_vectors.transpose())

    return feature_vectors


def create_cycle(n):
    g = Graph(directed=False)

    for i in range(n):
        g.add_vertex()

    for i in range(n-1):
        g.add_edge(i,i+1)

    g.add_edge(n-1,0)

    return g


def create_clique(n):
    g = Graph(directed=False)

    for i in range(n):
        g.add_vertex()

    for i in range(n):
        for j in range(i+1,n):
            g.add_edge(i,j)

    return g

subgraphs = []
for i in range(3,4):
    subgraphs.append(create_cycle(i))
#
# subgraphs = []
# for i in range(3,5):
#     subgraphs.append(create_clique(i))

ds = "PTC_FM" # 6-cycles.
ds = "PTC_MR" # 6-cycles.
ds = "MUTAG" # 6-cycles.
ds = "IMDB-MULTI"

# 1-WL.
gram_matrices = []
for i in range(1,6):
    print(i)
    graph_db, classes = read_txt(ds)
    gram_matrix = compute_wl(graph_db, i, compute_gram=False)
    gram_matrix = normalize_feature_vector_dense(gram_matrix)
    gram_matrices.append(gram_matrix)

acc, std, mrg = linear_svm_evaluation(gram_matrices, classes, num_repetitions=10)
print(acc, std, mrg)

print("###")
# 1-WL_F.
gram_matrices = []
for i in range(1,6):
    print(i)
    graph_db, classes = read_txt(ds)
    gram_matrix = compute_wl_f_all(graph_db, subgraphs, i, induced=True, compute_gram=False)
    gram_matrix = normalize_feature_vector_dense(gram_matrix)
    gram_matrices.append(gram_matrix)

acc, std, mrg = linear_svm_evaluation(gram_matrices, classes, num_repetitions=10)
print(acc, std, mrg)
print("#")
