import os
import sys

# allow python to find modules at the relative path
sys.path.insert(0, '/home/sami/repos/smbwebs/csap-graphpartitioning/src/python')
from scotch.graph_mapper import GraphMapper
from scotch.io import ScotchGraphArrays

SCOTCH_ENVIRONMENT_VALID = True # assume that the system is configured correctly for SCOTCH

# depending on scotch version that is available, SCOTCH_graphMapFixed may be available or not
SCOTCH_HAS_GRAPHMAPFIXED = True

# path to the scotch shared library
SCOTCH_LIB_PATH = ''

def init_scotch(pymodule_path, lib_path):
    # allow python to find modules at the relative path
    sys.path.insert(0, SCOTCH_PYLIB_REL_PATH)

    # try importing scotch
    try:
        from scotch.graph_mapper import GraphMapper
        from scotch.io import ScotchGraphArrays
        SCOTCH_LIB_PATH = libpath

    except ImportError as err:
        print(err)
        print("Could not load SCOTCH, check that SCOTCH_PYLIB_REL_PATH is set correctly in config.py")
        SCOTCH_ENVIRONMENT_VALID = False


def generate_prediction_model(graph,
                              num_iterations,
                              num_partitions,
                              assignments,
                              fixed):

    # SCOTCH algorithm\n",
    # we have networkx graph already, G
    scotchArrays = ScotchGraphArrays()
    scotchArrays.fromNetworkxGraph(graph, baseval=0)

    #scotchArrays.debugPrint()

    # create instance of SCOTCH
    mapper = GraphMapper('/usr/local/lib/libscotch.so')
    # set mapper parameters
    mapper.kbalval = 0.1
    mapper.numPartitions = num_partitions

    ok = mapper.initialize(scotchArrays, verbose=False)
    if(ok):
        # we can proceed with graphMap
        ok = mapper.graphMap()
        if(ok):
            assignments = mapper.scotchData._parttab
            print(assignments)
        else:
            print('Error while running graphMap()')
    else:
        print('Error while setting up SCOTCH for partitioning.')

