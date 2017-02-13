import numpy as np
import networkx as nx
from cpython cimport bool
from utils import bincount_assigned, score

cdef int UNMAPPED = -1
cdef bool DEBUG = False

class FennelPartitioner():

    def __init__(self, alpha=None):
        if alpha:
            self.PREDICTION_MODEL_ALPHA = alpha

    def get_votes(self, object graph, int node, int num_partitions, int[::] partition):
        global UNMAPPED
        seen = set()
        cdef float[::] partition_votes = np.zeros(num_partitions, dtype=np.float32)
        cdef int right_node = 0

        # find all neighbors from whole graph
        node_neighbors = list(nx.all_neighbors(graph, node))
        node_neighbors = [x for x in node_neighbors if x not in seen and not seen.add(x)]

        # calculate votes based on neighbors placed in partitions
        for right_node in node_neighbors:
            if partition[right_node] != UNMAPPED:
                partition_votes[partition[right_node]] += graph[node][right_node]['weight']
                #partition_votes_nodes[partition[right_node]] += 1

        return partition_votes

    def get_assignment(self,
                       object graph,
                       int node,
                       int num_partitions,
                       int[::] partition,
                       float[::] partition_votes,
                       float alpha):

        global DEBUG, UNMAPPED
        cdef int arg = 0
        cdef int max_arg = 0
        cdef float max_val = 0
        cdef float val = 0
        cdef int previous_assignment = 0

        assert partition is not None, "Blank partition passed"

        cdef float[::] partition_sizes = np.zeros(num_partitions, dtype=np.float32)
        s = bincount_assigned(graph, partition, num_partitions)
        partition_sizes = np.fromiter(s, dtype=np.float32)

        if DEBUG:
            print("Assigning node {}".format(node))
            print("\tPn = Votes - Alpha x Size")

        # Remember placement of node in the previous assignment
        previous_assignment = partition[node]

        max_arg = 0
        max_val = partition_votes[0] - alpha * partition_sizes[0]
        # XXX Loneliness score
        #if partition_votes[arg] > 0:
        #    max_val = 1 / partition_votes[arg]^n
        #else:
        #    max_val = 0

        if DEBUG:
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
            #if partition_votes[arg] > 0:
            #    val = 1 / partition_votes[arg]^n
            #else:
            #    val = 0

            if DEBUG:
                print("\tP{} = {} - {} x {} = {}".format(arg,
                                                         partition_votes[arg],
                                                         alpha,
                                                         partition_sizes[arg],
                                                         val))
            if previous_assignment == arg:
                # See comment above
                val += alpha
            if val > max_val: # XXX take account of partition_sizes
                max_arg = arg
                max_val = val

        if DEBUG:
            print("\tassigned to P{}".format(max_arg))

        return max_arg

    def fennel(self,
               object graph,
               int num_partitions,
               int[::] assignments,
               int[::] fixed,
               float alpha):

        global DEBUG, UNMAPPED
        cdef int node = 0

        single_nodes = []
        for node in graph.nodes_iter():

            # Exclude single nodes, deal with these later
            neighbors = list(nx.all_neighbors(graph, node))
            if not neighbors:
                single_nodes.append(node)
                continue

            # Skip fixed nodes
            if fixed[node] != UNMAPPED:
                if DEBUG:
                    print("Skipping node {}".format(node))
                continue

            partition_votes = self.get_votes(graph, node, num_partitions, assignments)
            assignments[node] = self.get_assignment(graph, node, num_partitions, assignments, partition_votes, alpha)

        # Assign single nodes
        node = 0
        for node in single_nodes:
            if assignments[node] == UNMAPPED:
                parts = bincount_assigned(graph, assignments, num_partitions)
                smallest = parts.index(min(parts))
                assignments[node] = smallest

        return np.asarray(assignments)

    def generate_prediction_model(self,
                                  object graph,
                                  int num_iterations,
                                  int num_partitions,
                                  int [::] assignments,
                                  int [::] fixed):

        cdef int i = 0

        for i in range(num_iterations):
            assignments = self.fennel(graph, num_partitions, assignments, fixed, self.PREDICTION_MODEL_ALPHA)

        return np.asarray(assignments)

