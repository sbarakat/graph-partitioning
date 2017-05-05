import os
import sys
import time # for PaToH's seed

import numpy as np
import networkx as nx

import graph_partitioning.utils as gputils
import graph_partitioning.partitioners.patoh.patoh as pat
import graph_partitioning.partitioners.patoh.patoh_data as patdata

class PatohPartitioner():
    def __init__(self, lib_path, quiet = True, partitioningIterations = 5, hyperedgeExpansionMode = 'no_expansion'):
        self.PATOH_LIB_PATH = lib_path
        self._quiet = quiet

        self.partitioningIterations = partitioningIterations
        self.hyperedgeExpansionMode = hyperedgeExpansionMode

        self.lib = pat.LibPatoh(self.PATOH_LIB_PATH)
        self.lib.load()

    def generate_prediction_model(self,
                                  graph,
                                  num_iterations,
                                  num_partitions,
                                  assignments,
                                  fixed):
        # STEP 0: sort the graph nodes
        sortedNodes = sorted(graph.nodes())

        # STEP 1: create a mapping of nodes for relabeling
        nodeMapping = {}
        for newID, nodeID in enumerate(sortedNodes):
            # old label as key, new label as value
            nodeMapping[nodeID] = newID

        # Create a new graph with the new mapping
        G = nx.relabel_nodes(graph, nodeMapping, copy=True)

        # Copy over the node and edge weightings: double check this
        for node in sortedNodes:
            newNode = nodeMapping[node]
            try:
                G.node[newNode]['weight'] = graph.node[node]['weight']

                for edge in graph.neighbors(node):
                    newEdge = nodeMapping[edge]
                    try:
                        G.edge[newNode][newEdge]['weight'] = graph.edge[node][edge]['weight']
                    except Exception as err:
                        pass
            except Exception as err:
                pass

        # Determine assignments
        patoh_assignments = np.full(G.number_of_nodes(), -1)
        for nodeID, assignment in enumerate(assignments):
            if nodeID in nodeMapping:
                # this nodeID is part of the mapping
                newNodeID = nodeMapping[nodeID]
                if fixed[nodeID] == 1:
                    patoh_assignments[newNodeID] = assignment

        #print('assignments', assignments)
        #print('fixed', fixed)
        #print('mapping', nodeMapping)
        #print('patoh_assignments', patoh_assignments)

        iterations = {}
        for i in range(0, self.partitioningIterations):
            _assignments = self._runPartitioning(G, num_partitions, patoh_assignments, nodeMapping, assignments)
            edges_cut, steps, cut_edges = gputils.base_metrics(graph, _assignments)
            if edges_cut not in list(iterations.keys()):
                iterations[edges_cut] = _assignments

        # return the minimum edges cut
        minEdgesCut = graph.number_of_edges()
        for key in list(iterations.keys()):
            if key < minEdgesCut:
                minEdgesCut = key

        assignments = iterations[minEdgesCut]

        self._printIterationStats(iterations)
        del iterations

        return assignments

    def _generate_prediction_model(self,
                                  graph,
                                  num_iterations,
                                  num_partitions,
                                  assignments,
                                  fixed):
        # STEP 0: sort the graph nodes
        gSortedNodes = sorted(graph.nodes())

        # create a mapping between the graph node ids and those used by SCOTCH
        # ensures nodes are numbered from 0...n-1
        node_indeces = self._createGraphIndeces(gSortedNodes, len(assignments))
        #print('node_indeces', node_indeces)

        # generate a new graph that only has the new nodes
        G = nx.Graph()
        for node in gSortedNodes:
            # set the new node index used by scotch
            G.add_node(node_indeces[node])
            try:
                G.node[node_indeces[node]]['weight'] = graph.node[node]['weight']
                #print("G.node[node_indeces[node]]['weight']", G.node[node_indeces[node]]['weight'])
            except Exception as err:
                pass

        # add the edges for each node using the new ids
        for node in gSortedNodes:
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

        iterations = {}
        for i in range(0, self.partitioningIterations):

            # perform an iteration of partitioning
            _assignments = self._runPartitioning(G, num_partitions, patoh_assignments, node_indeces, assignments)
            # compute the edges cut
            edges_cut, steps = gputils.base_metrics(graph, _assignments)
            if edges_cut not in list(iterations.keys()):
                iterations[edges_cut] = _assignments

        # return the minimum edges cut
        minEdgesCut = graph.number_of_edges()
        for key in list(iterations.keys()):
            if key < minEdgesCut:
                minEdgesCut = key

        assignments = iterations[minEdgesCut]

        self._printIterationStats(iterations)
        del iterations

        return assignments

    def _runPartitioning(self, G, num_partitions, patoh_assignments, node_indeces, input_assignments):
        # make a copy of arrays
        patoh_assignments_copy = np.array(patoh_assignments, copy=True).tolist()
        assignments = np.array(input_assignments, copy=True)

        # create the PaToH data
        patohdata = patdata.PatohData()
        patohdata.fromNetworkxGraph(G, num_partitions, partvec=patoh_assignments_copy, hyperedgeExpansionMode=self.hyperedgeExpansionMode)
        # initialize parameters
        ok = self.lib.initializeParameters(patohdata, num_partitions)
        if ok == False:
            print('Cannot Initialize PaToH parameters.')
            return assignments

        patohdata.params.seed = int(time.time() * 1000)
        #print('seed', patohdata.params.seed)
        #print(patohdata._nwghts)

        # check parameters
        if self.lib.checkUserParameters(patohdata, not self._quiet) == False:
            print('Error with PaToH parameters, check failed.')
            return assignments

        # alloc
        if self.lib.alloc(patohdata) == False:
            print('Error Allocating Memory for PaToH')
            return assignments

        # partition
        ok = self.lib.part(patohdata)
        if ok == True:
            #print('partvec', patohdata._partvec)
            # make a copy of the array
            #patoh_assignments = np.array(patohdata._partvec, copy=True)
            # re-map patoh_assignments back to assignments
            for oldNode in list(node_indeces.keys()):
                newNode = node_indeces[oldNode]
                #for oldNode, newNode in enumerate(node_indeces):
                assignments[oldNode] = patohdata._partvec[newNode]

        # free...
        self.lib.free(patohdata)
        del patohdata
        patohdata = None

        return assignments

    def _printIterationStats(self, iterations):
        if self._quiet == False:

            min_cuts = 1000000000
            max_cuts = 0

            for cuts in list(iterations.keys()):
                if cuts < min_cuts:
                    min_cuts = cuts
                if cuts > max_cuts:
                    max_cuts = cuts
            print('Ran PaToH for', self.partitioningIterations, 'iterations with min_cuts =', min_cuts, 'and max_cuts =', max_cuts, ' - picked min_cuts assignements.')

    def _createGraphIndeces(self, graphNodes, originalNodeNum):
        '''
            indeces[old_node_id] = new_node_id
        '''

        indeces = np.repeat(-1, originalNodeNum)
        nodeCount = 0
        for node in graphNodes:
            indeces[node] = nodeCount
            nodeCount += 1
        return indeces
