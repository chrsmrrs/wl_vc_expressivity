import numpy as np
from graph_tool.all import *
import graph_tool as gt

from auxiliarymethods.auxiliary_methods import read_txt
from auxiliarymethods.svm import kernel_svm_evaluation
from auxiliarymethods.svm import normalize_gram_matrix


# Simple implementation of 1-WL for edge and node-labeled graphs.
def compute_wl_f(graph_db, f, num_it, compute_gram):
    offset = 0
    graph_indices = []

    for g in graph_db:
        graph_indices.append((offset, offset + g.num_vertices() - 1))
        offset += g.num_vertices()

    # Compute 0th iteration.
    for i, g in enumerate(graph_db):
        for v in g.vertices():
            g.vp.nl[v] = 0

    # Label node according to f.
    for i, g in enumerate(graph_db):
        maps = gt.topology.subgraph_isomorphism(f, g, induced=True)
        for m in maps:
            for v in f.vertices():
                g.vp.nl[m[v]] = 1

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

                out_edges_v = g.get_out_edges(v).tolist()
                for (_, w) in out_edges_v:
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

    feature_vectors = np.array(feature_vectors)

    if compute_gram:
        feature_vectors = np.dot(feature_vectors, feature_vectors.transpose())

    return feature_vectors


# Simple implementation of 1-WL for edge and node-labeled graphs.
def compute_wl(graph_db, num_it, compute_gram):
    offset = 0
    graph_indices = []

    for g in graph_db:
        graph_indices.append((offset, offset + g.num_vertices() - 1))
        offset += g.num_vertices()

    # Compute 0th iteration.
    for i, g in enumerate(graph_db):
        for v in g.vertices():
            g.vp.nl[v] = 0

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

                out_edges_v = g.get_out_edges(v).tolist()
                for (_, w) in out_edges_v:
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

    feature_vectors = np.array(feature_vectors)

    if compute_gram:
        feature_vectors = np.dot(feature_vectors, feature_vectors.transpose())

    return feature_vectors


num_it = 6
ds_name = "ENZYMES"

gram_matrices = []
for i in range(num_it):
    graph_db, classes = read_txt(ds_name)
    gram_matrix = compute_wl(graph_db, num_it, compute_gram=True)
    gram_matrix = normalize_gram_matrix(gram_matrix)
    gram_matrices.append(gram_matrix)

acc, std = kernel_svm_evaluation(gram_matrices, classes)
print(acc, std)


# Create subgraph.
f = gt.Graph(directed=False)
a = f.add_vertex()
b = f.add_vertex()
c = f.add_vertex()
d = f.add_vertex()

f.add_edge(a,b)
f.add_edge(b,c)
f.add_edge(c,d)
f.add_edge(c,a)


gram_matrices = []
for i in range(num_it):
    graph_db, classes = read_txt(ds_name)
    gram_matrix = compute_wl_f(graph_db, f, num_it, compute_gram=True)
    gram_matrix = normalize_gram_matrix(gram_matrix)
    gram_matrices.append(gram_matrix)

acc, std = kernel_svm_evaluation([gram_matrix], classes)
print(acc, std)
