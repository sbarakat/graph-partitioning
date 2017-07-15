import numpy as np
import networkx as nx
from cpython cimport bool
from utils import bincount_assigned, score

cdef int UNMAPPED = -1
cdef bool DEBUG = False
cdef bool FRIEND_OF_FRIEND_ENABLED = False

class FennelPartitioner():

    def __init__(self, alpha=None):
        if alpha:
            self.PREDICTION_MODEL_ALPHA = alpha
        self.original_graph = None

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
                weight = graph.edge[node][right_node]['weight']
                if weight <= 0.0:
                    weight = 1.0
                partition_votes[partition[right_node]] += weight
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

            # Skip fixed nodes - check this first and move on if node has already been considered
            if fixed[node] != UNMAPPED:
                if DEBUG:
                    print("Skipping node {}".format(node))
                continue

            # Exclude single nodes, deal with these later
            neighbors = list(nx.all_neighbors(graph, node))
            if not neighbors:
                single_nodes.append(node)
                continue

            partition_votes = self.get_votes(graph, node, num_partitions, assignments)
            assignments[node] = self.get_assignment(graph, node, num_partitions, assignments, partition_votes, alpha)

        # Assign single nodes that have no neighbors
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

        current_batch = []
        if FRIEND_OF_FRIEND_ENABLED:
            current_batch = self.current_batch_nodes(graph, fixed)

        for i in range(num_iterations):
            assignments = self.fennel(graph, num_partitions, assignments, fixed, self.PREDICTION_MODEL_ALPHA)

        if FRIEND_OF_FRIEND_ENABLED:
            # compute improved assignment for everyone in the current batch that has no friends in a partition
            self.friend_of_friend_lonely_node_partition_assignment(graph, num_partitions, current_batch, assignments, fixed)

        return np.asarray(assignments)

    def current_batch_nodes(self, object graph, int [::] fixed):
        '''
        Returns the list of nodes in graph that currently haven't been assigned to any single partition just yet
        '''
        current_batch = []
        cdef int node = 0
        for node in graph.nodes_iter():
            if fixed[node] == UNMAPPED:
                current_batch.append(node)
        return current_batch

    def friend_of_friend_lonely_node_partition_assignment(self,
                                  object graph,
                                  int num_partitions,
                                  list current_batch_nodes,
                                  int [::] assignments,
                                  int [::] fixed):
        '''
        This function determines the optimal partition for a node that is lonely.

1. for each fennel batch, I run fennel with all of its iterations, which produces the assignments
2. once that is done, i make a list of nodes, from the latest batch that have just been assigned and from that list, I extract the nodes that have no friends in any of the partitions, the lonely_nodes list
2.a the way the lonely nodes are found is: i call get_votes() from FENNEL, which, given a node, makes the sum of edge weights between that node and all nodes in each partition. so a partition with score 0.0 = a partition where this node has no edges, so if the sum of partition_votes = 0 across all partitions, then that node has no friends in any partition
3. for each lonely node, i perform the following
3.a find all the neighbours of that node, whether they have been assigned already or not, i then remove the neighbours that are already in a partition (a case that should never happen given 2a above)
3.b for each neighbour, I compute the partition_scores for that neighbour, as if I were trying to assign it now, given the partition score, i remove any neighbour that has NO friends currently assigned by fennel
3.c at this point, the neighbour has at least a friend in any partition, so I compute the FENNEL assignment, calling get_assignment() which assigns that node to their hypothetical partition
3.d i store a count of where friends of the lonely node get partitioned: example 'lonely_node friend counts', 50, {2: 3, 0: 2}. This means that node 50 is a lonely_node and 3 of its friends have been assigned by fennel to partition 2 and 2 of its friends have been assigned to partition 0
3.e at this point, I should relocate node 50 from its currently assigned partition, to partition 2

MISSING: node relocation out of current assignment to find neighbor scores and then relocation after scores computed

        '''

        #print('STARTING friend_of_friend_lonely_node_partition_assignment')

        # obtain list of friendless nodes
        lonely_nodes = []
        for node in current_batch_nodes:
            # check how many friends are assigned to partitions for the current node and current subgraph
            partition_scores = list(self.get_votes(graph, node, num_partitions, assignments))
            if self.node_has_friends_in_partitions(partition_scores) == False:
                # check if node has any edges at all in whole graph
                lonely_nodes.append(node)

        #print('lonely nodes:', lonely_nodes)

        for lonely_node in lonely_nodes:
            neighbors = []

            if self.original_graph:
                # use the full graph
                neighbors = list(nx.all_neighbors(self.original_graph, lonely_node))
            else:
                neighbors = list(nx.all_neighbors(graph, lonely_node))

            #print('lonely_node_neighbors', lonely_node, neighbors)

            if len(neighbors) > 0:
                # this is a single node, a random assignment is OK
                #print('Found FENNEL node that has no friends in partitions, but that has some friends in network', lonely_node, neighbors)
                friend_count_per_partition = {}
                # check where these neighbors are more likely to end up
                #cdef int neighbor = 0
                # on a first pass over the neighbors, compute the total partition scores and move the lonely_node to that partition
                total_neighbor_partition_scores = {}
                neighbor_partition_scores = {}
                original_partition = assignments[lonely_node]
                assignments[lonely_node] = -1

                for neighbor in neighbors:
                    if fixed[neighbor] != UNMAPPED:
                        continue # neighbor is in network - should never be the case

                    if fixed[neighbor] != UNMAPPED:
                        continue # neighbor is in network - should never be the case
                    neighbor_partition_scores[neighbor] = self.get_votes(self.original_graph, neighbor, num_partitions, assignments)
                    for i, score in enumerate(list(neighbor_partition_scores[neighbor])):
                        if i in total_neighbor_partition_scores:
                            total_neighbor_partition_scores[i] += score
                        else:
                            total_neighbor_partition_scores[i] = score

                # move lonely node to partition with highest neighbor score
                max_score = 0.0
                max_score_partition = -1
                for partition in total_neighbor_partition_scores.keys():
                    if total_neighbor_partition_scores[partition] > max_score:
                        max_score_partition = partition
                        max_score = total_neighbor_partition_scores[partition]

                # relocate node
                if(max_score_partition >= 0):
                    #print('relocating lonely_node', lonely_node, original_partition, max_score_partition, total_neighbor_partition_scores)
                    assignments[lonely_node] = max_score_partition
                else:
                    assignments[lonely_node] = original_partition

                for neighbor in neighbors:
                    if fixed[neighbor] != UNMAPPED:
                        continue # neighbor is in network - should never be the case
                    #partition_scores = self.get_votes(self.original_graph, neighbor, num_partitions, assignments)
                    if self.node_has_friends_in_partitions(list(neighbor_partition_scores[neighbor])) == True:
                    #if self.node_has_friends_in_partitions(list(partition_scores)) == True:
                        # this neighbor has at least one friend in any one of the partitions
                        # determine in which partition this friend would end up
                        partition = self.get_assignment(self.original_graph, neighbor, num_partitions, assignments, neighbor_partition_scores[neighbor], self.PREDICTION_MODEL_ALPHA)
                        if partition in friend_count_per_partition:
                            friend_count_per_partition[partition] += 1
                        else:
                            friend_count_per_partition[partition] = 1
                    #print('friend', neighbor, list(partition_scores))
                #print('lonely_node friend counts', lonely_node, friend_count_per_partition)
                # relocate lonely_node based on friend_count_per_partition
                max_count = 0
                best_partition = -1
                for partition in friend_count_per_partition.keys():
                    if friend_count_per_partition[partition] > max_count:
                        max_count = friend_count_per_partition[partition]
                        best_partition = partition

                if best_partition >= 0:
                    #print('relocating lonely_node to best partition', lonely_node, max_count, best_partition)
                    assignments[lonely_node] = best_partition

    def node_has_friends_in_partitions(self, partition_scores):
        sum_scores = 0.0
        for score in partition_scores:
            sum_scores += score
        if sum_scores == 0.0:
            return False # has no friends in any partition
        return True # has some friends in at least one partition
