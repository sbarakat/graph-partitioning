import os
import shared
import argparse
import networkx as nx

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--data-file", "-i", metavar='F', default="",
                    help="Input data file")
parser.add_argument("--output-dir", "-o", metavar='F', default="",
                    help="Output data directory")
parser.add_argument("--num-partitions", metavar='N', type=int, default=4,
                    help="Number of shelters.")
parser.add_argument("--num-iterations", metavar='N', type=int, default=10,
                    help="The number of iterations when making prediction model.")
parser.add_argument("--prediction-model-cut-off", metavar='N', type=float, default=0.10,
                    help="Percentage of prediction model to use before discarding. When set to 0, prediction model is discarded, useful for one-shot.")
parser.add_argument("--one-shot-alpha", metavar='N', type=float, default=0.5,
                    help="Alpha value used in one-shot (when --restream-batches set to 1)")
parser.add_argument("--restream-batches", metavar='N', type=int, default=10,
                    help="Number of arrivals to batch before recalculating alpha and restreaming.")
parser.add_argument("--use-virtual-nodes", action="store_true", default=False,
                    help="Create virtual nodes based on prediction model.")
parser.add_argument("--virtual-node-weight", metavar='N', type=float, default=1.0,
                    help="Virtual nodes: node weight.")
parser.add_argument("--virtual_edge_weight", metavar='N', type=float, default=1.0,
                    help="Virtual nodes: edge weight.")
args = parser.parse_args()


# coding: utf-8

# In[1]:

#pwd = get_ipython().magic('pwd')

#DATA_FILENAME = os.path.join(pwd, "data", "oneshot_fennel_weights.txt")
#OUTPUT_DIRECTORY = os.path.join(pwd, "output")

DATA_FILENAME = args.data_file
OUTPUT_DIRECTORY = args.output_dir

# Read input file for prediction model, if not provided a prediction
# model is made using FENNEL
PREDICTION_MODEL = ""

# File containing nodes that need a shelter and ones that don't. Nodes
# represented by line number; value of 1 represents a shelter is needed;
# value of 0 represents shelter is not needed.
#NEEDS_SHELTER_FILE = os.path.join(pwd, "data", "needs_shelter.txt")
NEEDS_SHELTER_FILE = ""

# Number of shelters
num_partitions = args.num_partitions

# The number of iterations when making prediction model
num_iterations = args.num_iterations

# Percentage of prediction model to use before discarding
# When set to 0, prediction model is discarded, useful for one-shot
prediction_model_cut_off = args.prediction_model_cut_off

# Alpha value used in one-shot (when restream_batches set to 1)
one_shot_alpha = args.one_shot_alpha

# Number of arrivals to batch before recalculating alpha and restreaming.
# When set to 1, one-shot is used with alpha value from above
restream_batches = args.restream_batches

# Create virtual nodes based on prediction model
use_virtual_nodes = args.use_virtual_nodes

# Virtual nodes: node weight and edge weight
virtual_node_weight = 1.0
virtual_edge_weight = 1.0

# read METIS file
G, edge_weights = shared.read_metis(DATA_FILENAME)

# Order of nodes arriving
arrival_order = list(range(0, G.number_of_nodes()))
#random.shuffle(arrival_order)

# If set, the node weight is set to 100 if the node requires a shelter, otherwise 0.
adjust_node_weight_to_needs_shelter = False

# Alpha value used in prediction model
prediction_model_alpha = G.number_of_edges() * (num_partitions / G.number_of_nodes()**2)

if NEEDS_SHELTER_FILE == "":
    # mark all nodes as needing a shelter
    needs_shelter = [1]*G.number_of_nodes()
else:
    with open(NEEDS_SHELTER_FILE, "r") as f:
        needs_shelter = [int(line.rstrip('\n')) for line in f]

print("Graph loaded...")
print("Nodes: {}".format(G.number_of_nodes()))
print("Edges: {}".format(G.number_of_edges()))
if nx.is_directed(G):
    print("Graph is directed")
else:
    print("Graph is undirected")


# In[2]:

# Set high node weight for those that need a shelter, and reduce for those that don't
if adjust_node_weight_to_needs_shelter:
    for i, s in enumerate(needs_shelter):
        n = i+1
        if s == 1:
            G.node[n]['weight'] = 100
        else:
            G.node[n]['weight'] = 0

# Update edge weights for nodes that have an assigned probability of displacement
for edge in G.edges_iter(data=True):
    left = edge[0]
    right = edge[1]
    edge_weight = edge[2]['weight']

    # new edge weight
    edge[2]['weight'] = (float(G.node[left]['weight']) * edge_weight) * (float(G.node[right]['weight']) * edge_weight)


# In[3]:

get_ipython().magic('load_ext Cython')
#get_ipython().magic('pylab inline')


# In[4]:

get_ipython().run_cell_magic('cython', '', 'import numpy as np\nimport networkx as nx\nfrom shared import bincount_assigned\n\ncdef int UNMAPPED = -1\n\ndef get_votes(graph, int node, float[::] edge_weights, int num_partitions, int[::] partition):\n    seen = set()\n    cdef float[::] partition_votes = np.zeros(num_partitions, dtype=np.float32)\n\n    # find all neighbors from whole graph\n    node_neighbors = list(nx.all_neighbors(graph, node))\n    node_neighbors = [x for x in node_neighbors if x not in seen and not seen.add(x)]\n\n    # calculate votes based on neighbors placed in partitions\n    for n in node_neighbors:\n        if partition[n] != UNMAPPED:\n            partition_votes[partition[n]] += edge_weights[n]\n            \n    return partition_votes\n\ndef get_assignment(graph,\n                   int node,\n                   int num_partitions,\n                   int[::] partition,\n                   float[::] partition_votes,\n                   float alpha,\n                   int debug):\n\n    cdef int arg = 0\n    cdef int max_arg = 0\n    cdef float max_val = 0\n    cdef float val = 0\n    cdef int previous_assignment = 0\n\n    assert partition is not None, "Blank partition passed"\n\n    cdef float[::] partition_sizes = np.zeros(num_partitions, dtype=np.float32)\n    s = bincount_assigned(graph, partition, num_partitions)\n    partition_sizes = np.fromiter(s, dtype=np.float32)\n    \n    if debug:\n        print("Assigning node {}".format(node))\n        print("\\tPn = Votes - Alpha x Size")\n\n    # Remember placement of node in the previous assignment\n    previous_assignment = partition[node]\n\n    max_arg = 0\n    max_val = partition_votes[0] - alpha * partition_sizes[0]\n    if debug:\n        print("\\tP{} = {} - {} x {} = {}".format(0,\n                                                 partition_votes[0],\n                                                 alpha,\n                                                 partition_sizes[0],\n                                                 max_val))\n\n    if previous_assignment == 0:\n        # We remove the node from its current partition before\n        # deciding to re-add it, so subtract alpha to give\n        # result of 1 lower partition size.\n        max_val += alpha\n\n    for arg in range(1, num_partitions):\n        val = partition_votes[arg] - alpha * partition_sizes[arg]\n\n        if debug:\n            print("\\tP{} = {} - {} x {} = {}".format(arg,\n                                                     partition_votes[arg],\n                                                     alpha,\n                                                     partition_sizes[arg],\n                                                     val))\n        if previous_assignment == arg:\n            # See comment above\n            val += alpha\n        if val > max_val:\n            max_arg = arg\n            max_val = val\n\n    if debug:\n        print("\\tassigned to P{}".format(max_arg))\n\n    return max_arg\n\ndef fennel_rework(graph, \n                  float[::] edge_weights,\n                  int num_partitions,\n                  int[::] assignments,\n                  int[::] fixed,\n                  float alpha,\n                  int debug):\n\n    single_nodes = []\n    for n in graph.nodes_iter():\n\n        # Exclude single nodes, deal with these later\n        neighbors = list(nx.all_neighbors(graph, n))\n        if not neighbors:\n            single_nodes.append(n)\n            continue\n            \n        # Skip fixed nodes\n        if fixed[n] != UNMAPPED:\n            if debug:\n                print("Skipping node {}".format(n))\n            continue\n\n        partition_votes = get_votes(graph, n, edge_weights, num_partitions, assignments)\n        assignments[n] = get_assignment(graph, n, num_partitions, assignments, partition_votes, alpha, debug)\n\n    # Assign single nodes\n    for n in single_nodes:\n        if assignments[n] == UNMAPPED:\n            parts = bincount_assigned(graph, assignments, num_partitions)\n            smallest = parts.index(min(parts))\n            assignments[n] = smallest\n\n    return np.asarray(assignments)')


# In[5]:

UNMAPPED = -1

# reset
assignments = np.repeat(np.int32(UNMAPPED), G.number_of_nodes())
fixed = np.repeat(np.int32(UNMAPPED), G.number_of_nodes())

print("PREDICTION MODEL")
print("----------------\n")
print("WASTE\t\tCUT RATIO\tMISMATCH")

if PREDICTION_MODEL:
    with open(PREDICTION_MODEL, "r") as inf:
        assignments = np.fromiter(inf.readlines(), dtype=np.int32)
    x = shared.score(G, assignments)
    print("{0:.5f}\t\t{1:.10f}\t{2}".format(x[0], x[1], x[2]))

else:
    for i in range(num_iterations):
        alpha = prediction_model_alpha
        assignments = fennel_rework(G, edge_weights, num_partitions, assignments, fixed, alpha, 0)

        x = shared.score(G, assignments)
        print("{0:.5f}\t\t{1:.10f}\t{2}".format(x[0], x[1], x[2]))

print("\nAssignments:")
shared.fixed_width_print(assignments)

nodes_fixed = len([o for o in fixed if o == 1])
print("\nFixed: {}".format(nodes_fixed))

shared.print_partitions(G, assignments, num_partitions)


# In[6]:

if use_virtual_nodes:
    print("Creating virtual nodes and assigning edges based on prediction model")

    # create virtual nodes
    virtual_nodes = list(range(G.number_of_nodes(), G.number_of_nodes() + num_partitions))
    print("\nVirtual nodes:")
    print(virtual_nodes)

    # create virtual edges
    virtual_edges = []
    for n in range(0, G.number_of_nodes()):
        virtual_edges += [(n, virtual_nodes[assignments[n]])]

    # extend assignments
    assignments = np.append(assignments, np.array(list(range(0, num_partitions)), dtype=np.int32))
    fixed = np.append(fixed, np.array([1] * num_partitions, dtype=np.int32))

    G.add_nodes_from(virtual_nodes, weight=virtual_node_weight)
    G.add_edges_from(virtual_edges, weight=virtual_edge_weight)

    edge_weights = np.array([x[2]['weight'] for x in G.edges(data=True)], dtype=np.float32)

    print("\nAssignments:")
    shared.fixed_width_print(assignments)
    print("Last {} nodes are virtual nodes.".format(num_partitions))


# In[7]:

cut_off_value = int(prediction_model_cut_off * G.number_of_nodes())
if prediction_model_cut_off == 0:
    print("Discarding prediction model\n")
else:
    print("Assign first {} arrivals using prediction model, then discard\n".format(cut_off_value))

# fix arrivals
nodes_arrived = []
print("WASTE\t\tCUT RATIO\tMISMATCH")
for a in arrival_order:
    # check if node needs a shelter
    if needs_shelter[a] == 0:
        continue

    nodes_fixed = len([o for o in fixed if o == 1])
    if nodes_fixed >= cut_off_value:
        break
    fixed[a] = 1
    nodes_arrived.append(a)

    # make a subgraph of all arrived nodes
    Gsub = G.subgraph(nodes_arrived)

    x = shared.score(Gsub, assignments, num_partitions)
    print("{0:.5f}\t\t{1:.10f}\t{2}".format(x[0], x[1], x[2]))

# remove nodes not fixed, ie. discard prediction model
for i in range(0, len(assignments)):
    if fixed[i] == -1:
        assignments[i] = -1

print("WASTE\t\tCUT RATIO\tMISMATCH")
x = shared.score(G, assignments, num_partitions)
print("{0:.5f}\t\t{1:.10f}\t{2}".format(x[0], x[1], x[2]))

print("\nAssignments:")
shared.fixed_width_print(assignments)

nodes_fixed = len([o for o in fixed if o == 1])
print("\nFixed: {}".format(nodes_fixed))

shared.print_partitions(G, assignments, num_partitions)


# In[8]:

if restream_batches == 1:
    print("One-shot assignment mode")
    print("------------------------\n")
else:
    print("Re-streaming in batches of {}".format(restream_batches))
    print("--------------------------------\n")

batch_arrived = []
print("WASTE\t\tCUT RATIO\tMISMATCH\tALPHA")
for a in arrival_order:
    # check if node needs a shelter
    if needs_shelter[a] == 0:
        continue

    # check if node is already arrived
    if fixed[a] == 1:
        continue

    # one-shot assigment: assign each node as it arrives
    if restream_batches == 1:
        alpha = one_shot_alpha
        partition_votes = get_votes(G, a, edge_weights, num_partitions, assignments)
        assignments[a] = get_assignment(G, a, num_partitions, assignments, partition_votes, alpha, 0)
        fixed[a] = 1
        nodes_arrived.append(a)

        # make a subgraph of all arrived nodes
        Gsub = G.subgraph(nodes_arrived)

        x = shared.score(Gsub, assignments, num_partitions)
        print("{0:.5f}\t\t{1:.10f}\t{2}\t\t{3:.10f}".format(x[0], x[1], x[2], alpha))
        continue

    batch_arrived.append(a)

    if restream_batches == len(batch_arrived):

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

        # restream
        for n in batch_arrived:
            partition_votes = get_votes(Gsub, n, edge_weights, num_partitions, assignments)
            assignments[n] = get_assignment(G, n, num_partitions, assignments, partition_votes, alpha, 0)
            fixed[n] = 1
            nodes_arrived.append(n)

        x = shared.score(Gsub, assignments, num_partitions)
        print("{0:.5f}\t\t{1:.10f}\t{2}\t\t{3:.10f}".format(x[0], x[1], x[2], alpha))
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


# In[9]:

if use_virtual_nodes:
    print("Remove virtual nodes")
    
    print("\nCurrent graph:")
    print("Nodes: {}".format(G.number_of_nodes()))
    print("Edges: {}".format(G.number_of_edges()))

    G.remove_nodes_from(virtual_nodes)
    assignments = np.delete(assignments, virtual_nodes)
    fixed = np.delete(fixed, virtual_nodes)

    print("\nVirtual nodes removed:")
    print("Nodes: {}".format(G.number_of_nodes()))
    print("Edges: {}".format(G.number_of_edges()))


# In[10]:

# Add partition attribute to nodes
for i in range(0, len(assignments)):
    G.add_nodes_from([i], partition=str(assignments[i]))

# Freeze Graph from further modification
G = nx.freeze(G)


# In[11]:

import os
import datetime

timestamp = datetime.datetime.now().strftime('%H%M%S')
data_filename,_ = os.path.splitext(os.path.basename(DATA_FILENAME))
data_filename += "-" + timestamp

graph_metrics = {
    "file": timestamp,
    "num_partitions": num_partitions,
    "num_iterations": num_iterations,
    "prediction_model_cut_off": prediction_model_cut_off,
    "one_shot_alpha": one_shot_alpha,
    "restream_batches": restream_batches,
    "use_virtual_nodes": use_virtual_nodes,
    "virtual_node_weight": virtual_node_weight,
    "virtual_edge_weight": virtual_edge_weight,
}
graph_fieldnames = [
    "file",
    "num_partitions",
    "num_iterations",
    "prediction_model_cut_off",
    "one_shot_alpha",
    "restream_batches",
    "use_virtual_nodes",
    "virtual_node_weight",
    "virtual_edge_weight",
    "edges_cut",
    "waste",
    "cut_ratio",
    "communication_volume",
    "network_permanence",
    "Q",
    "NQ",
    "Qds",
    "intraEdges",
    "interEdges",
    "intraDensity",
    "modularity degree",
    "conductance",
    "expansion",
    "contraction",
    "fitness",
    "QovL",
]

print("Complete graph with {} nodes".format(G.number_of_nodes()))
(file_maxperm, file_oslom) = shared.write_graph_files(OUTPUT_DIRECTORY, "{}-all".format(data_filename), G)

# original scoring algorithm
scoring = shared.score(G, assignments, num_partitions)
graph_metrics.update({
    "waste": scoring[0],
    "cut_ratio": scoring[1],
})

# edges cut and communication volume
edges_cut, steps = shared.base_metrics(G)
graph_metrics.update({
    "edges_cut": edges_cut,
    "communication_volume": steps,
})

# MaxPerm
max_perm = shared.run_max_perm(file_maxperm)
graph_metrics.update({"network_permanence": max_perm})

# Community Quality metrics
community_metrics = shared.run_community_metrics(OUTPUT_DIRECTORY,
                                                 "{}-all".format(data_filename),
                                                 file_oslom)
graph_metrics.update(community_metrics)

print("\nConfig")
print("-------\n")
for f in graph_fieldnames[:9]:
    print("{}: {}".format(f, graph_metrics[f]))

print("\nMetrics")
print("-------\n")
for f in graph_fieldnames[9:]:
    print("{}: {}".format(f, graph_metrics[f]))

# write metrics to CSV
metrics_filename = os.path.join(OUTPUT_DIRECTORY, "metrics.csv")
shared.write_metrics_csv(metrics_filename, graph_fieldnames, graph_metrics)


# In[12]:

partition_metrics = {}
partition_fieldnames = [
    "file",
    "partition",
    "network_permanence",
    "Q",
    "NQ",
    "Qds",
    "intraEdges",
    "interEdges",
    "intraDensity",
    "modularity degree",
    "conductance",
    "expansion",
    "contraction",
    "fitness",
    "QovL",
]

for p in range(0, num_partitions):
    partition_metrics = {
        "file": timestamp,
        "partition": p
    }

    nodes = [i for i,x in enumerate(assignments) if x == p]
    Gsub = G.subgraph(nodes)
    print("\nPartition {} with {} nodes".format(p, Gsub.number_of_nodes()))
    print("-----------------------------\n")

    (file_maxperm, file_oslom) = shared.write_graph_files(OUTPUT_DIRECTORY, "{}-p{}".format(data_filename, p), Gsub)
    
    # MaxPerm
    max_perm = shared.run_max_perm(file_maxperm)
    partition_metrics.update({"network_permanence": max_perm})

    # Community Quality metrics
    community_metrics = shared.run_community_metrics(OUTPUT_DIRECTORY,
                                                     "{}-p{}".format(data_filename, p),
                                                     file_oslom)
    partition_metrics.update(community_metrics)

    print("\nMetrics")
    for f in partition_fieldnames:
        print("{}: {}".format(f, partition_metrics[f]))

    # write metrics to CSV
    metrics_filename = os.path.join(OUTPUT_DIRECTORY, "metrics-partitions.csv")
    shared.write_metrics_csv(metrics_filename, partition_fieldnames, partition_metrics)


# In[ ]:



