import numpy as np
import networkx as nx
from cpython cimport bool
from shared import bincount_assigned, score

cdef int UNMAPPED = -1

def get_votes(graph, int node, int num_partitions, int[::] partition):
    seen = set()
    cdef float[::] partition_votes = np.zeros(num_partitions, dtype=np.float32)

    # find all neighbors from whole graph
    node_neighbors = list(nx.all_neighbors(graph, node))
    node_neighbors = [x for x in node_neighbors if x not in seen and not seen.add(x)]

    # calculate votes based on neighbors placed in partitions
    for right_node in node_neighbors:
        if partition[right_node] != UNMAPPED:
            partition_votes[partition[right_node]] += graph[node][right_node]['weight']

    return partition_votes

def get_assignment(graph,
                   int node,
                   int num_partitions,
                   int[::] partition,
                   float[::] partition_votes,
                   float alpha,
                   int debug):

    cdef int arg = 0
    cdef int max_arg = 0
    cdef float max_val = 0
    cdef float val = 0
    cdef int previous_assignment = 0

    assert partition is not None, "Blank partition passed"

    cdef float[::] partition_sizes = np.zeros(num_partitions, dtype=np.float32)
    s = bincount_assigned(graph, partition, num_partitions)
    partition_sizes = np.fromiter(s, dtype=np.float32)

    if debug:
        print("Assigning node {}".format(node))
        print("\tPn = Votes - Alpha x Size")

    # Remember placement of node in the previous assignment
    previous_assignment = partition[node]

    max_arg = 0
    max_val = partition_votes[0] - alpha * partition_sizes[0]
    if debug:
        print("\tP{} = {} - {} x {} = {}".format(0,
                                                 partition_votes[0],
                                                 alpha,
                                                 partition_sizes[0],
                                                 max_val))

    if previous_assignment == 0:
        # We remove the node from its current partition before
        # deciding to re-add it, so subtract alpha to give
        # result of 1 lower partition size.
        max_val += alpha

    for arg in range(1, num_partitions):
        val = partition_votes[arg] - alpha * partition_sizes[arg]

        if debug:
            print("\tP{} = {} - {} x {} = {}".format(arg,
                                                     partition_votes[arg],
                                                     alpha,
                                                     partition_sizes[arg],
                                                     val))
        if previous_assignment == arg:
            # See comment above
            val += alpha
        if val > max_val:
            max_arg = arg
            max_val = val

    if debug:
        print("\tassigned to P{}".format(max_arg))

    return max_arg

def fennel(graph,
           int num_partitions,
           int[::] assignments,
           int[::] fixed,
           float alpha,
           int debug):

    single_nodes = []
    for n in graph.nodes_iter():

        # Exclude single nodes, deal with these later
        neighbors = list(nx.all_neighbors(graph, n))
        if not neighbors:
            single_nodes.append(n)
            continue

        # Skip fixed nodes
        if fixed[n] != UNMAPPED:
            if debug:
                print("Skipping node {}".format(n))
            continue

        partition_votes = get_votes(graph, n, num_partitions, assignments)
        assignments[n] = get_assignment(graph, n, num_partitions, assignments, partition_votes, alpha, debug)

    # Assign single nodes
    for n in single_nodes:
        if assignments[n] == UNMAPPED:
            parts = bincount_assigned(graph, assignments, num_partitions)
            smallest = parts.index(min(parts))
            assignments[n] = smallest

    return np.asarray(assignments)

def generate_prediction_model(graph,
                              int num_iterations,
                              int num_partitions,
                              int [::] assignments,
                              int [::] fixed,
                              float alpha):

    for i in range(num_iterations):
        assignments = fennel(graph, num_partitions, assignments, fixed, alpha, 0)

    return np.asarray(assignments)
