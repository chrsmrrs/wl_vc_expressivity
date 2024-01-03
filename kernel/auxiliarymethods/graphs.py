import graph_tool as gt
from graph_tool.all import *

# Create cycle on n vertices.
def create_cycle(n):
    g = Graph(directed=False)

    for i in range(n):
        g.add_vertex()

    for i in range(n - 1):
        g.add_edge(i, i + 1)

    g.add_edge(n - 1, 0)

    return g

# Create clique on n vertices.
def create_clique(n):
    g = Graph(directed=False)

    for i in range(n):
        g.add_vertex()

    for i in range(n):
        for j in range(i+1,n):
            g.add_edge(i,j)

    return g










