import os
import sys

import numpy as np
import networkx as nx

import graph_partitioning.partitioners.patoh.patoh as pat
import graph_partitioning.partitioners.patoh.patoh_data as patdata

class PatohPartitioner():
    def __init__(self, lib_path, quiet = True):
        self.PATOH_LIB_PATH = lib_path
        self._quiet = quiet

        self.lib = pat.LibPatoh(self.PATOH_LIB_PATH)
        self.lib.load()

    def generate_prediction_model(self,
                                  graph,
                                  num_iterations,
                                  num_partitions,
                                  assignments,
                                  fixed):
        # create a mapping between the graph node ids and those used by SCOTCH
        # ensures nodes are numbered from 0...n-1
        node_indeces = self._createGraphIndeces(graph, len(assignments))
        #print('node_indeces', node_indeces)

        # generate a new graph that only has the new nodes
        G = nx.Graph()
        for node in graph.nodes():
            # set the new node index used by scotch
            G.add_node(node_indeces[node])
            try:
                G.node[node_indeces[node]]['weight'] = graph.node[node]['weight']
            except Exception as err:
                pass

        # add the edges for each node using the new ids
        for node in graph.nodes():
            newNodeID = node_indeces[node]
            for edge in graph.neighbors(node):
                newEdgeID = node_indeces[edge]
                G.add_edge(newNodeID, newEdgeID)
                try:
                    weight = graph[node][edge]['weight']
                    G[newNodeID][newEdgeID]['weight'] = weight
                except Exception as err:
                    pass

        # determine the assignment partitions
        patoh_assignments = []
        for nodeID, assignment in enumerate(assignments):
            if node_indeces[nodeID] >= 0:
                # this nodeID is part of this graph and needs to be partitioned
                # add node's fixed partition, if present
                patoh_assignments.append(assignment)

        patohdata = patdata.PatohData()
        patohdata.fromNetworkxGraph(G, num_partitions, partvec=patoh_assignments)

        # read hypergraph... (should be OK above)

        # initialize parameters
        ok = self.lib.initializeParameters(patohdata, num_partitions)
        if ok == False:
            # TODO throw exception...?
            print('Cannot Initialize PaToH parameters.')
            return assignments

        # check parameters
        if self.lib.checkUserParameters(patohdata, not self._quiet) == False:
            print('Error with PaToH parameters.')
            return assignments

        # alloc
        if self.lib.alloc(patohdata) == False:
            print('Error Allocating Memory for PaToH')
            return assignments

        # partition
        ok = self.lib.part(patohdata)
        if ok == True:
            # make a copy of the array
            #patoh_assignments = np.array(patohdata._partvec, copy=True)
            # re-map patoh_assignments back to assignments
            for oldNode, newNode in enumerate(node_indeces):
                assignments[oldNode] = patohdata._partvec[newNode]

        # free...
        self.lib.free(patohdata)
        patohdata = None

        return assignments

    def _createGraphIndeces(self, graph, originalNodeNum):
        '''
            indeces[old_node_id] = new_node_id
        '''

        indeces = np.repeat(-1, originalNodeNum)
        nodeCount = 0
        for node in graph.nodes():
            indeces[node] = nodeCount
            nodeCount += 1
        return indeces
