import os
import sys

import numpy as np
import networkx as nx

from scotch.graph_mapper import GraphMapper
from scotch.io import ScotchGraphArrays

import utilities.alg_utils as algutils

'''
TASK: check neighbors for each node and add virtual edges
TASK: update node mapping for valid scotch usable graph

Set parttab to assignments (for already fixed nodes already)
'''


class ScotchPartitioner():

    def __init__(self, lib_path, virtualNodesEnabled = False):
        self.SCOTCH_LIB_PATH = lib_path
        self.virtualNodesEnabled = virtualNodesEnabled

    def _generate_prediction_model(self,
                                  graph,
                                  num_iterations,
                                  num_partitions,
                                  assignments,
                                  fixed):
        # Simple version not using graph map fixed.

        # SCOTCH algorithm
        # we have networkx graph already, G
        scotchArrays = ScotchGraphArrays()
        scotchArrays.fromNetworkxGraph(graph, baseval=0)

        #scotchArrays.debugPrint()

        # create instance of SCOTCH
        mapper = GraphMapper(self.SCOTCH_LIB_PATH)

        # set mapper parameters
        mapper.kbalval = 0.1
        mapper.numPartitions = num_partitions

        ok = mapper.initialize(scotchArrays, verbose=False)
        if ok:
            # we can proceed with graphMap
            ok = mapper.graphMap()
            if ok:
                return mapper.scotchData._parttab

            else:
                print('Error while running graphMap()')
        else:
            print('Error while setting up SCOTCH for partitioning.')

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


        # if there are edgeless nodes, then we need virtual nodes
        requires_virtual = self._requiresVirtualNodes(graph)
        virtual_nodes = []
        if requires_virtual:
            # add virtual nodes to the new graph G
            virtual_nodes = self._createVirtualNodes(G, num_partitions)
            #print('virtual_nodes', virtual_nodes)

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

        # add all the virtual edges
        virtual_edges = {}
        if requires_virtual:
            virtual_edges = self._virtualEdges(graph, assignments, num_partitions, virtual_nodes)
            for key in list(virtual_edges.keys()):
                newID = node_indeces[key]
                G.add_edge(newID, virtual_edges[key])
                #print('virtual_edge', newID, virtual_edges[key])

        # determine the assignment partitions
        scotch_assignments = []
        for nodeID, assignment in enumerate(assignments):
            if node_indeces[nodeID] >= 0:
                # this nodeID is part of this graph and needs to be partitioned
                # add node's fixed partition, if present
                scotch_assignments.append(assignment)

            '''
            if nodeID in list(virtual_edges.keys()):
                # fix this node
                scotch_assignments[node_indeces[nodeID]] = virtual_nodes.index(virtual_edges[nodeID])
                print('fixing', nodeID, node_indeces[nodeID], virtual_nodes.index(virtual_edges[nodeID]))
            '''

        # add virtual nodes to assignments
        if requires_virtual:
            for i in range(0, num_partitions):
                scotch_assignments.append(i)

        # SCOTCH algorithm
        # we have networkx graph already, G
        scotchArrays = ScotchGraphArrays()
        scotchArrays.fromNetworkxGraph(G, parttab=scotch_assignments, baseval=0)

        #print('parttab', scotchArrays._parttab)
        #scotchArrays.debugPrint()

        # create instance of SCOTCH
        mapper = GraphMapper(self.SCOTCH_LIB_PATH)

        # set mapper parameters
        mapper.kbalval = 0.1
        mapper.numPartitions = num_partitions

        ok = mapper.initialize(scotchArrays, verbose=False)
        if ok:
            # we can proceed with graphMap
            ok = mapper.graphMapFixed()
            if ok:
                scotch_assignments = mapper.scotchData._parttab
                if requires_virtual:
                    # remove the virtual nodes from assignments
                    for virtualN in virtual_nodes:
                        #id = len(scotch_assignments) - 1
                        #del scotch_assignments[-1]
                        G.remove_node(virtualN)

                # update assignments
                for oldNode, newNode in enumerate(node_indeces):
                    assignments[oldNode] = scotch_assignments[newNode]
                '''
                for node in G.nodes():
                    originalNode = node_indeces.index(node)
                    #originalNode = graph.nodes()[node]
                    assignments[originalNode] = scotch_assignments[node]
                    #print('setting', node, originalNode, scotch_assignments[node])
                '''
                return assignments
            else:
                print('Error while running graphMap()')
        else:
            print('Error while setting up SCOTCH for partitioning.')

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

    def _requiresVirtualNodes(self, graph):
        if (self.virtualNodesEnabled == False):
            # we don't allow virtual nodes
            return False

        for node in graph.nodes():
            if len(graph.neighbors(node)) == 0:
                #print(node, 'has no neighbors')
                return True
        return False

    def _createVirtualNodes(self, graph, num_partitions):
        virtual_nodes = []
        for i in range(0, num_partitions):
            virtualNode = graph.number_of_nodes()
            graph.add_node(virtualNode)
            virtual_nodes.append(virtualNode)
        return virtual_nodes

    def _virtualEdges(self, graph, assignments, num_partitions, virtual_nodes):
        virtual_edges = {}
        tmp, partitions = algutils.minPartitionCounts(assignments, num_partitions)
        for node in graph.nodes():
            if len(graph.neighbors(node)) == 0:
                #print(node, ' has no neighbors')
                # this node has no neighbors, choose a node in a partition
                #partition, partitions = algutils.minPartitionCounts(assignments, num_partitions)

                #print('partitions', partitions)
                minPart = 1000000
                _partition = -1
                for partition in partitions:
                    if partitions[partition] == 0:
                        # pick this
                        _partition = partition
                        #partitions[partition] += 1
                        break
                    elif partitions[partition] < minPart:
                        minPart = partitions[partition]
                        #partitions[partition] += 1
                        _partition = partition
                partitions[_partition] += 1
                virtualNode = virtual_nodes[_partition]

                #graph.add_edge(virtualNode, node)
                virtual_edges[node] = virtualNode
        return virtual_edges
