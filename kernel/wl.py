import graph_tool as gt
import numpy as np
from graph_tool.all import *


# Simple implementation of 1-WL_F.
def compute_wl_f(graph_db, f_list, num_it, induced, compute_gram):
    num_subgraphs = len(f_list)
    offset = 0
    graph_indices = []
    color_manager = {}
    c = 0

    # Manage labels and indices.
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

    # Label node according to subgraphs.
    for i, g in enumerate(graph_db):
        # Iterate over subgraphs.
        for s, f in enumerate(f_list):
            # Compute subgraph isomorphisms from f to g.
            maps = gt.topology.subgraph_isomorphism(f, g, induced=induced)

            for m in maps:
                for v in f.vertices():
                    g.vp.labels[m[v]][s] += 1

    # Compress vector labels.
    for i, g in enumerate(graph_db):
        for v in g.vertices():
            h = hash(tuple(g.vp.labels[v]))

            if h in color_manager:
                g.vp.nl[v] = color_manager[h]
            else:
                color_manager[h] = c
                g.vp.nl[v] = c
                c += 1

    # Compute 1-WL over node-labeled graphs.
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

    # Comptue Gram matrix.
    if compute_gram:
        feature_vectors = np.dot(feature_vectors, feature_vectors.transpose())

    return feature_vectors



# Simple implementation of 1-WL_F.
def compute_wloa_f(graph_db, f_list, num_it, induced):
    num_graphs = len(graph_db)
    num_subgraphs = len(f_list)
    offset = 0
    graph_indices = []
    color_manager = {}
    c = 0

    # Manage labels and indices.
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

    # Label node according to subgraphs.
    for i, g in enumerate(graph_db):
        # Iterate over subgraphs.
        for s, f in enumerate(f_list):
            # Compute subgraph isomorphisms from f to g.
            maps = gt.topology.subgraph_isomorphism(f, g, induced=induced)

            for m in maps:
                for v in f.vertices():
                    g.vp.labels[m[v]][s] += 1

    # Compress vector labels.
    for i, g in enumerate(graph_db):
        for v in g.vertices():
            h = hash(tuple(g.vp.labels[v]))

            if h in color_manager:
                g.vp.nl[v] = color_manager[h]
            else:
                color_manager[h] = c
                g.vp.nl[v] = c
                c += 1

    # Compute 1-WL over node-labeled graphs.
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

    gram_matrix = np.zeros([num_graphs,num_graphs], dtype=np.double)
    for i in range(num_graphs):
        for j in range(i+1,num_graphs):
            v = feature_vectors[[i, j], :].min(axis=0).sum()
            gram_matrix[i][j] += v
            gram_matrix[j][i] += v
    return gram_matrix

# Simple implementation of 1-WL.
def compute_wl(graph_db, num_it, compute_gram):
    offset = 0
    graph_indices = []

    # Manage labels and indices.
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

    # Comptue Gram matrix.
    if compute_gram:
        feature_vectors = np.dot(feature_vectors, feature_vectors.transpose())

    return feature_vectors


# Simple implementation of 1-WL.
def compute_wloa(graph_db, num_it):
    num_graphs = len(graph_db)

    offset = 0
    graph_indices = []

    # Manage labels and indices.
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

    gram_matrix = np.zeros([num_graphs,num_graphs], dtype=np.double)
    for i in range(num_graphs):
        for j in range(i+1,num_graphs):
            v = feature_vectors[[i, j], :].min(axis=0).sum()
            gram_matrix[i][j] += v
            gram_matrix[j][i] += v
    return gram_matrix
