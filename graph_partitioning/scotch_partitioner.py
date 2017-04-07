import os
import sys

import numpy as np
import networkx as nx

import graph_partitioning.partitioners.utils as putils
import graph_partitioning.partitioners.scotch.scotch as scotch
import graph_partitioning.partitioners.scotch.scotch_data as sdata

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
        # STEP 0: sort the graph nodes
        sortedNodes = sorted(graph.nodes())

        # STEP 1: create a mapping of nodes for relabeling
        nodeMapping = {}
        for newID, nodeID in enumerate(sortedNodes):
            # old label as key, new label as value
            nodeMapping[nodeID] = newID

        #print(nodeMapping)

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
        scotch_assignments = np.full(G.number_of_nodes(), -1)
        for nodeID, assignment in enumerate(assignments):
            if nodeID in nodeMapping:
                # this nodeID is part of the mapping
                newNodeID = nodeMapping[nodeID]
                if fixed[nodeID] == 1:
                    scotch_assignments[newNodeID] = assignment

        #print('G.nodes', G.nodes())
        #print('scotch_assignments', scotch_assignments)

        # SCOTCH algorithm
        # Load the graph into the SCOTCH array structures
        scotchArrays = sdata.ScotchData()
        scotchArrays.fromNetworkxGraph(G, parttab=scotch_assignments, baseval=0)

        #scotchArrays.debugPrint()

        # create instance of SCOTCH Library
        mapper = scotch.Scotch(self.SCOTCH_LIB_PATH)

        # set the mapper parameters
        mapper.kbalval = 0.01
        mapper.numPartitions = num_partitions

        ok = mapper.initialize(scotchArrays, verbose=False)
        if ok:
            # we can proceed with graphMap
            print('gmf')
            ok = mapper.graphMapFixed()
            print('gmfend')
            if ok:
                scotch_assignments = mapper.scotchData._parttab
                # update assignments
                for oldNode in list(nodeMapping.keys()):
                    newNode = nodeMapping[oldNode]
                    assignments[oldNode] = scotch_assignments[newNode]
                print('returning')
                return assignments
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
        # STEP 0: sort the graph nodes
        gSortedNodes = sorted(graph.nodes())

        # STEP 1: map between graph nodes and SCOTCH nodes
        # create a mapping between the graph node ids and those used by SCOTCH
        # ensures that nodes are numbered from 0...n-1 for SCOTCH especially when some nodes in graph have been fixed
        node_indeces = self._createGraphIndeces(gSortedNodes, len(assignments))

        # generate a new graph that only has the new nodes
        G = nx.Graph()
        for node in gSortedNodes:
            # set the new node index used by scotch
            G.add_node(node_indeces[node])
            try:
                # set the node weight
                G.node[node_indeces[node]]['weight'] = graph.node[node]['weight']
            except Exception as err:
                pass

        # STEP 2: add virtual nodes, if enabled and required
        # if there are edgeless nodes, then we need virtual nodes - this may actually not be needed
        requires_virtual = self._requiresVirtualNodes(graph)
        virtual_nodes = []
        if requires_virtual:
            # add virtual nodes to the new graph G
            virtual_nodes = self._createVirtualNodes(G, num_partitions)

        # STEP 3: add edges & weights using the new ID mapping
        # add the edges for each node using the new ids
        for node in gSortedNodes:
            newNodeID = node_indeces[node]
            for edge in graph.neighbors(node):
                newEdgeID = node_indeces[edge]
                G.add_edge(newNodeID, newEdgeID)
                try:
                    weight = graph.edge[node][edge]['weight']
                    G.edge[newNodeID][newEdgeID]['weight'] = weight
                except Exception as err:
                    pass

        # STEP 4: add virtual edges where needed
        virtual_edges = {}
        if requires_virtual:
            virtual_edges = self._virtualEdges(graph, assignments, num_partitions, virtual_nodes)
            for key in list(virtual_edges.keys()):
                newID = node_indeces[key]
                G.add_edge(newID, virtual_edges[key])

        # determine the nodes that are already assigned to their respective partition
        scotch_assignments = []
        for nodeID, assignment in enumerate(assignments):
            if node_indeces[nodeID] >= 0:
                # this nodeID is part of this graph and needs to be partitioned
                # add node's fixed partition, if present
                scotch_assignments.append(assignment)

        #print('lenass', len(scotch_assignments), G.number_of_nodes())
        #print('assignments', assignments)
        # add virtual nodes to assignments
        if requires_virtual:
            for i in range(0, num_partitions):
                scotch_assignments.append(i)

        #node_weights = {n[0]: n[1]['weight'] for n in G.nodes_iter(data=True)}
        #print('scotchnw', node_weights)

        # SCOTCH algorithm
        # Load the graph into the SCOTCH array structures
        scotchArrays = sdata.ScotchData()
        scotchArrays.fromNetworkxGraph(G, parttab=scotch_assignments, baseval=0)

        #scotchArrays.debugPrint()

        # create instance of SCOTCH Library
        mapper = scotch.Scotch(self.SCOTCH_LIB_PATH)

        # set the mapper parameters
        mapper.kbalval = 0.001
        mapper.numPartitions = num_partitions

        ok = mapper.initialize(scotchArrays, verbose=False)
        if ok:
            # we can proceed with graphMap
            print('pre_partitioned_Ass', mapper.scotchData._parttab)
            print('edgewhts', mapper.scotchData._edlotab)
            ok = mapper.graphMapFixed()
            if ok:
                print('partitioned_Ass', mapper.scotchData._parttab)
                scotch_assignments = mapper.scotchData._parttab
                if requires_virtual:
                    #print('requires_virtual')
                    # remove the virtual nodes from assignments
                    for virtualN in virtual_nodes:
                        G.remove_node(virtualN)

                # update assignments
                for oldNode, newNode in enumerate(node_indeces):
                    if(newNode >= 0):
                        #aold = assignments[oldNode]
                        assignments[oldNode] = scotch_assignments[newNode]
                        #print(oldNode, newNode, aold, scotch_assignments[newNode], assignments[oldNode])
                return assignments
            else:
                print('Error while running graphMap()')
        else:
            print('Error while setting up SCOTCH for partitioning.')

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
        tmp, partitions = putils.minPartitionCounts(assignments, num_partitions)
        for node in graph.nodes():
            if len(graph.neighbors(node)) == 0:
                #print(node, ' has no neighbors')
                # this node has no neighbors, choose a node in a partition
                #partition, partitions = putils.minPartitionCounts(assignments, num_partitions)

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
