

def generate_prediction_model():

    # SCOTCH algorithm\n",
    # we have networkx graph already, G
    scotchArrays = ScotchGraphArrays()
    scotchArrays.fromNetworkxGraph(G, baseval=0)

    #scotchArrays.debugPrint()

    # create instance of SCOTCH
    mapper = GraphMapper(config.SCOTCH_LIB_PATH)
    # set mapper parameters
    mapper.kbalval = 0.1
    mapper.numPartitions = num_partitions

    ok = mapper.initialize(scotchArrays, verbose=False)
    if(ok):
        # we can proceed with graphMap
        ok = mapper.graphMap()
        if(ok):
            assignments = mapper.scotchData._parttab
            #print(assignments)
        else:
            print('Error while running graphMap()')
    else:
        print('Error while setting up SCOTCH for partitioning.')

