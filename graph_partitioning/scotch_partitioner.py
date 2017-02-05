import os
import sys

from scotch.graph_mapper import GraphMapper
from scotch.io import ScotchGraphArrays

class ScotchPartitioner():

    def __init__(self, lib_path):
        self.SCOTCH_LIB_PATH = lib_path

    def generate_prediction_model(self,
                                  graph,
                                  num_iterations,
                                  num_partitions,
                                  assignments,
                                  fixed):

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
                assignments = mapper.scotchData._parttab
                return assignments
            else:
                print('Error while running graphMap()')
        else:
            print('Error while setting up SCOTCH for partitioning.')

