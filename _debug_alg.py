import os
import shared
import networkx as nx
import numpy as np

import pyximport; pyximport.install()
import fennel

# ******** SECTION 1 ************* #

try:
    import config
except ImportError as err:
    print(err)
    print("**Could not load config.py\n**Copy config_template.py and rename it.")

pwd = os.getcwd()

DATA_FILENAME = os.path.join(pwd, "data", "oneshot_fennel_weights.txt")
OUTPUT_DIRECTORY = os.path.join(pwd, "output")

# Read input file for prediction model, if not provided a prediction
# model is made using FENNEL
PREDICTION_MODEL = ""

# File containing simulated arrivals. This is used in simulating nodes
# arriving at the shelter. Nodes represented by line number; value of
# 1 represents a node as arrived; value of 0 represents the node as not
# arrived or needing a shelter.
SIMULATED_ARRIVAL_FILE = os.path.join(pwd, "data", "simulated_arrival.txt")
#SIMULATED_ARRIVAL_FILE = ""

# File containing the geographic location of each node.
POPULATION_LOCATION_FILE = os.path.join(pwd, "data", "population_location.csv")

# Number of shelters
num_partitions = 4

# The number of iterations when making prediction model
num_iterations = 10

# Percentage of prediction model to use before discarding
# When set to 0, prediction model is discarded, useful for one-shot
prediction_model_cut_off = 0.10

# Alpha value used in one-shot (when restream_batches set to 1)
one_shot_alpha = 0.5

# Number of arrivals to batch before recalculating alpha and restreaming.
# When set to 1, one-shot is used with alpha value from above
restream_batches = 10

# Create virtual nodes based on prediction model
use_virtual_nodes = False

# Virtual nodes: edge weight
virtual_edge_weight = 1.0


####
# GRAPH MODIFICATION FUNCTIONS

# Also enables the edge calculation function.
graph_modification_functions = True

# If set, the node weight is set to 100 if the node arrives at the shelter,
# otherwise the node is removed from the graph.
alter_arrived_node_weight_to_100 = True

# Uses generalized additive models from R to generate prediction of nodes not
# arrived. This sets the node weight on unarrived nodes the the prediction
# given by a GAM.
# Needs POPULATION_LOCATION_FILE to be set.
alter_node_weight_to_gam_prediction = True

gam_k_value = 100

# Alter the edge weight for nodes that haven't arrived. This is a way to
# de-emphasise the prediction model for the unknown nodes.
prediction_model_emphasis = 1.0

# read METIS file
G = shared.read_metis(DATA_FILENAME)

# Alpha value used in prediction model
prediction_model_alpha = G.number_of_edges() * (num_partitions / G.number_of_nodes()**2)

# Order of nodes arriving
arrival_order = list(range(0, G.number_of_nodes()))

# Arrival order should not be shuffled if using GAM to alter node weights
#random.shuffle(arrival_order)

if SIMULATED_ARRIVAL_FILE == "":
    # mark all nodes as needing a shelter
    simulated_arrival_list = [1]*G.number_of_nodes()
else:
    with open(SIMULATED_ARRIVAL_FILE, "r") as ar:
        simulated_arrival_list = [int(line.rstrip('\n')) for line in ar]

print("Graph loaded...")
print("Nodes: {}".format(G.number_of_nodes()))
print("Edges: {}".format(G.number_of_edges()))
if nx.is_directed(G):
    print("Graph is directed")
else:
    print("Graph is undirected")


# ******** END SECTION 1 ********** #

# ******** SECTION 2 **************#

# setup for other algorithms
if config.ENABLE_SCOTCH == True:
    # import the relevant SCOTCH modules
    from scotch.graph_mapper import GraphMapper
    from scotch.io import ScotchGraphArrays

UNMAPPED = -1

# reset
assignments = np.repeat(np.int32(UNMAPPED), G.number_of_nodes())
fixed = np.repeat(np.int32(UNMAPPED), G.number_of_nodes())

print("PREDICTION MODEL")
print("----------------")

# Display which algorithm is being run
if config.PREDICTION_MODEL_ALGORITHM == config.Partitioners.FENNEL:
    print("Using: FENNEL Partitioning")
    print("---------------\n")
elif config.PREDICTION_MODEL_ALGORITHM == config.Partitioners.SCOTCH:
    print("Using: SCOTCH Partitioning")
    print("--------------------------\n")

predictionModels = {}
# store model data for different types of partitioners
# NOTE: THIS IS NOT IMPLEMENTED YET - need to discuss first
if config.RUN_ALL_PREDICTION_MODEL_ALGORITHMS == True:
    # create different prediction models
    fennelModel = {}
    fennelModel['assignments'] = np.repeat(np.int32(UNMAPPED), G.number_of_nodes())
    fennelModel['fixed'] = np.repeat(np.int32(UNMAPPED), G.number_of_nodes())

    predictionModels[config.Partitioners.FENNEL] = fennelModel

    scotchModel = {}
    scotchModel['assignments'] = np.repeat(np.int32(UNMAPPED), G.number_of_nodes())
    scotchModel['fixed'] = np.repeat(np.int32(UNMAPPED), G.number_of_nodes())

    predictionModels[config.Partitioners.SCOTCH] = scotchModel

# Begin computation of prediction model
if PREDICTION_MODEL:
    # if we have a prediction model from file, load it
    with open(PREDICTION_MODEL, "r") as inf:
        assignments = np.fromiter(inf.readlines(), dtype=np.int32)

else:
    # choose the right algorithm
    if config.PREDICTION_MODEL_ALGORITHM == config.Partitioners.FENNEL:
        assignments = fennel.generate_prediction_model(G, num_iterations, num_partitions, assignments, fixed, prediction_model_alpha)
    elif config.PREDICTION_MODEL_ALGORITHM == config.Partitioners.SCOTCH:
        # SCOTCH algorithm
        # we have a networkx graph already, G
        scotchArrays = ScotchGraphArrays() # create the object storing all the SCOTCH arrays
        scotchArrays.fromNetworkxGraph(G, baseval=0) # populate arrays from G

        #scotchArrays.debugPrint() # uncomment this to print out contents of scotchArrays

        # create instance of SCOTCH Library
        mapper = GraphMapper(config.SCOTCH_LIB_PATH)

        # set some optional parameters for the SCOTCH_Arch, SCOTCH_Strat, SCOTCH_Graph
        # see csap-graphpartitioning/src/python/scotch/graph_mapper: GraphMapper.__init__() method for more options
        mapper.kbalval = 0.1
        mapper.numPartitions = num_partitions

        # intializes the SCOTCH_Arch, SCOTCH_Strat, SCOTCH_Graph using scotchArray and optional parameters
        ok = mapper.initialize(scotchArrays, verbose=False)
        if(ok):
            # we can proceed with graphMap, the data structures were setup correctly
            ok = mapper.graphMap()
            if(ok):
                # graphMap was run successfully, copy the assignments
                # make a deep copy as we then delete the mapper data, to clear memory
                # and the array reference may be lost
                assignments = np.array(mapper.scotchData._parttab, copy=True)

                mapper.delObjects()
            else:
                print('Error while running graphMap()')
        else:
            print('Error while setting up SCOTCH for partitioning.')

x = shared.score(G, assignments, num_partitions)
edges_cut, steps = shared.base_metrics(G, assignments)
print("WASTE\t\tCUT RATIO\tEDGES CUT\tCOMM VOLUME")
print("{0:.5f}\t\t{1:.10f}\t{2}\t\t{3}".format(x[0], x[1], edges_cut, steps))

print("\nAssignments:")
shared.fixed_width_print(assignments)

nodes_fixed = len([o for o in fixed if o == 1])
print("\nFixed: {}".format(nodes_fixed))

shared.print_partitions(G, assignments, num_partitions)

# ******** END SECTION 2 ********** #

# ******** START SECTION 3 ********** #

if use_virtual_nodes:
    print("Creating virtual nodes and assigning edges based on prediction model")

    # create virtual nodes
    virtual_nodes = list(range(G.number_of_nodes(), G.number_of_nodes() + num_partitions))
    print("\nVirtual nodes:")

    # create virtual edges
    virtual_edges = []
    for n in range(0, G.number_of_nodes()):
        virtual_edges += [(n, virtual_nodes[assignments[n]])]

    # extend assignments
    assignments = np.append(assignments, np.array(list(range(0, num_partitions)), dtype=np.int32))
    fixed = np.append(fixed, np.array([1] * num_partitions, dtype=np.int32))

    G.add_nodes_from(virtual_nodes, weight=1)
    G.add_edges_from(virtual_edges, weight=virtual_edge_weight)

    print("\nAssignments:")
    shared.fixed_width_print(assignments)
    print("Last {} nodes are virtual nodes.".format(num_partitions))

# ******** END SECTION 3 ********** #

# ******** START SECTION 4 ********** #

cut_off_value = int(prediction_model_cut_off * G.number_of_nodes())
if prediction_model_cut_off == 0:
    print("Discarding prediction model\n")
else:
    print("Assign first {} arrivals using prediction model, then discard\n".format(cut_off_value))

# fix arrivals
nodes_arrived = []
for a in arrival_order:
    # check if node needs a shelter
    if simulated_arrival_list[a] == 0:
        continue

    # set 100% node weight for those that need a shelter
    if alter_arrived_node_weight_to_100:
        G.node[a]['weight'] = 100

    nodes_fixed = len([o for o in fixed if o == 1])
    if nodes_fixed >= cut_off_value:
        break
    fixed[a] = 1
    nodes_arrived.append(a)

# remove nodes not fixed, ie. discard prediction model
for i in range(0, len(assignments)):
    if fixed[i] == -1:
        assignments[i] = -1

x = shared.score(G, assignments, num_partitions)
edges_cut, steps = shared.base_metrics(G, assignments)
print("WASTE\t\tCUT RATIO\tEDGES CUT\tCOMM VOLUME")
print("{0:.5f}\t\t{1:.10f}\t{2}\t\t{3}".format(x[0], x[1], edges_cut, steps))

print("\nAssignments:")
shared.fixed_width_print(assignments)

nodes_fixed = len([o for o in fixed if o == 1])
print("\nFixed: {}".format(nodes_fixed))

shared.print_partitions(G, assignments, num_partitions)

# ******** END SECTION 4 ********** #

# ******** START SECTION 5 ********** #

if restream_batches == 1:
    print("One-shot assignment mode")
    print("------------------------\n")
else:
    print("Assigning in batches of {}".format(restream_batches))
    print("--------------------------------\n")

def edge_expansion(G):
    # Update edge weights for nodes that have an assigned probability of displacement
    for edge in G.edges_iter(data=True):
        left = edge[0]
        right = edge[1]
        edge_weight = edge[2]['weight_orig']

        # new edge weight
        edge[2]['weight'] = (float(G.node[left]['weight']) * edge_weight) * (float(G.node[right]['weight']) * edge_weight)

        if left in nodes_arrived or right in nodes_arrived:
            # change the emphasis of the prediction model
            edge[2]['weight'] = edge[2]['weight'] * prediction_model_emphasis

    return G

# preserve original node/edge weight
if graph_modification_functions:
    node_weights = {n[0]: n[1]['weight'] for n in G.nodes_iter(data=True)}
    nx.set_node_attributes(G, 'weight_orig', node_weights)

    edge_weights = {(e[0], e[1]): e[2]['weight'] for e in G.edges_iter(data=True)}
    nx.set_edge_attributes(G, 'weight_orig', edge_weights)


# SETUP SCOTCH VARIABLES
scotchMapper = None
scotchArrayData = None
if config.ASSIGNMENT_MODEL_ALGORITHM == config.Partitioners.SCOTCH:
    scotchMapper = GraphMapper(config.SCOTCH_LIB_PATH, numPartitions=num_partitions)
    scotchArrayData = ScotchGraphArrays()

# FOR DEBUGGING PURPOSES:
print('Graph mod fncs:',graph_modification_functions)
print('restream_batches:', restream_batches)

batch_arrived = []
print("WASTE\t\tCUT RATIO\tEDGES CUT\tCOMM VOLUME\tALPHA")
for i, a in enumerate(arrival_order):

    # check if node is already arrived
    if fixed[a] == 1:
        continue

    # GRAPH MODIFICATION FUNCTIONS
    if graph_modification_functions:

        # remove nodes that don't need a shelter
        if simulated_arrival_list[a] == 0:
            print('Removing Node', a)
            G.remove_node(a)
            continue

        # set 100% node weight for those that need a shelter
        if alter_arrived_node_weight_to_100:
            print("Setting weight=100 on node", a)
            G.node[a]['weight'] = 100

    # one-shot assigment: assign each node as it arrives
    if restream_batches == 1:
        alpha = one_shot_alpha

        if config.ASSIGNMENT_MODEL_ALGORITHM == config.Partitioners.FENNEL:
            partition_votes = fennel.get_votes(G, a, num_partitions, assignments)
            assignments[a] = fennel.get_assignment(G, a, num_partitions, assignments, partition_votes, alpha, 0)
        elif config.ASSIGNMENT_MODEL_ALGORITHM == config.Partitioners.SCOTCH:
            # load array data from graph
            scotchArrayData.fromNetworkxGraph(G, parttab=assignments)
            ok = scotchMapper.initialize(scotchArrayData)
            if(ok):
                # mapper initialized
                ok = scotchMapper.graphMapFixed()
                if(ok):
                    assignments = scotchMapper.scotchData._parttab
                else:
                    print("Error running graphMapFixed()")
            else:
                print("Error initializing SCOTCH GraphMapper for graphMapFixed()")
        fixed[a] = 1
        nodes_arrived.append(a)

        # make a subgraph of all arrived nodes
        Gsub = G.subgraph(nodes_arrived)

        x = shared.score(Gsub, assignments, num_partitions)
        edges_cut, steps = shared.base_metrics(Gsub, assignments)
        print("{0:.5f}\t\t{1:.10f}\t{2}\t\t{3}\t\t{4:.10f}".format(x[0], x[1], edges_cut, steps, alpha))
        continue

    batch_arrived.append(a)

    # NOTE: TEMPORARY -> enable graph_modification_functions
    graph_modification_functions = False

    if restream_batches == len(batch_arrived) or i == len(arrival_order) - 1:

        # GRAPH MODIFICATION FUNCTIONS
        if graph_modification_functions:

            # set node weight to prediction generated from a GAM
            if alter_node_weight_to_gam_prediction:
                total_arrived = nodes_arrived + batch_arrived + [a]
                if len(total_arrived) < gam_k_value:
                    k = len(total_arrived)
                else:
                    k = gam_k_value

                gam_weights = shared.gam_predict(POPULATION_LOCATION_FILE, len(total_arrived), k)

                for node in G.nodes_iter():
                    if alter_arrived_node_weight_to_100 and node in total_arrived:
                        pass # weight would have been set previously
                    else:
                        G.node[node]['weight'] = int(gam_weights[node] * 100)

            G = edge_expansion(G)

        # make a subgraph of all arrived nodes
        Gsub = G.subgraph(nodes_arrived + batch_arrived)

        # recalculate alpha
        if Gsub.is_directed():
            # as it's a directed graph, edges_arrived is actually double, so divide by 2
            edges_arrived = Gsub.number_of_edges() / 2
        else:
            edges_arrived = Gsub.number_of_edges()
        nodes_fixed = len([o for o in fixed if o == 1])
        alpha = (edges_arrived) * (num_partitions / (nodes_fixed + len(batch_arrived))**2)

        if alter_node_weight_to_gam_prediction:
            # justification: the gam learns the entire population, so run fennal on entire population
            assignments = fennel.generate_prediction_model(G,
                                                           num_iterations,
                                                           num_partitions,
                                                           assignments,
                                                           fixed,
                                                           alpha)
        else:
            # use the information we have, those that arrived
            assignments = fennel.generate_prediction_model(Gsub,
                                                           num_iterations,
                                                           num_partitions,
                                                           assignments,
                                                           fixed,
                                                           alpha)


        # assign nodes to prediction model
        for n in batch_arrived:
            fixed[n] = 1
            nodes_arrived.append(n)

        x = shared.score(Gsub, assignments, num_partitions)
        edges_cut, steps = shared.base_metrics(Gsub, assignments)
        print("{0:.5f}\t\t{1:.10f}\t{2}\t\t{3}\t\t{4:.10f}".format(x[0], x[1], edges_cut, steps, alpha))
        batch_arrived = []

# remove nodes not fixed
for i in range(0, len(assignments)):
    if fixed[i] == -1:
        assignments[i] = -1

print("\nAssignments:")
shared.fixed_width_print(assignments)

nodes_fixed = len([o for o in fixed if o == 1])
print("\nFixed: {}".format(nodes_fixed))

shared.print_partitions(G, assignments, num_partitions)
# ******** END SECTION 5 ********** #
