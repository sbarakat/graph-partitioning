
import argparse

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

DATA_FILENAME = args.data_file
OUTPUT_DIRECTORY = args.output_dir

# Read input file for prediction model
PREDICTION_MODEL = ""

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

# Go to cell 3 to shuffle arrivals


# In[2]:

import numpy as np
import networkit
import networkx as nx

# Reading data
# - neither networkit nor networkx handle node weights
# - networkit can read the METIS file format, networkx can't
# - networkit does not support extra attributes to nodes or
#    edges, however they can be added later when writing to
#    a GraphML file format[1]
# - networkx support node and edge attributes, so we can keep
#    the partition assignment with the node and also support
#    virtual nodes without needing to maintain a seperate
#    data structure.
# - the most sensible method for loading the graph data is to
#    read the METIS file with networkit, convert the graph to
#    a networkx graph, then read the METIS file once again
#    and load the node weights into a networkx node attribute
#
# Writing data
# - to be able to write the output data with the partition
#    each node is assigned to, a suitable file format to write
#    to is needed
# - writing to a METIS file will lose the partition assignments
# - if we use networkit to write the data, then the only function
#    available is GraphMLWriter()
# - networkx provides a richer set of output methods which
#    preserve the partition assignment
# - using networkit to write GML data causes a loss of edge weights and node weights
# - using networkx to write GML data preserves node and edge weights
# [1]: https://networkit.iti.kit.edu/data/uploads/docs/NetworKit-Doc/python/html/graphio.html#networkit.graphio.GraphMLWriter

# read METIS file
print("Loading graph data...")
nkG = networkit.graphio.METISGraphReader().read(DATA_FILENAME)

# convert to networkx Graph
G = networkit.nxadapter.nk2nx(nkG)

# add node weights from METIS file
with open(DATA_FILENAME, "r") as metis:
    
    # read meta data from first line
    first_line = next(metis).split()
    m_nodes = int(first_line[0])
    m_edges = int(first_line[1])

    for i, line in enumerate(metis):
        if line.strip():
            weight = line.split()[0]
            G.add_nodes_from([i], weight=str(weight))
        else:
            # blank line indicates no node weight
            G.add_nodes_from([i], weight=0.0)

edges = np.array(G.edges(), dtype=np.int32)
edge_weights = np.array([x[2]['weight'] for x in G.edges(data=True)], dtype=np.float32)
node_weights = np.array([x[1]['weight'] for x in G.nodes(data=True)], dtype=np.float32)

# sanity check
assert (m_nodes == G.number_of_nodes())
assert (m_nodes == len(node_weights))
assert (m_edges == G.number_of_edges())
assert (m_edges == len(edge_weights))
assert (m_edges == len(edges))

print("Nodes: {}".format(G.number_of_nodes()))
print("Edges: {}".format(G.number_of_edges()))
if nx.is_directed(G):
    print("Graph is directed")
else:
    print("Graph is undirected")


# In[3]:

# Order of people arriving
arrivals = list(range(0, G.number_of_nodes()))
#random.shuffle(arrivals)

# Alpha value used in prediction model
prediction_model_alpha = G.number_of_edges() * (num_partitions / G.number_of_nodes()**2)


# In[4]:

get_ipython().magic('load_ext Cython')
#get_ipython().magic('pylab inline')


# In[5]:

get_ipython().run_cell_magic('cython', '', 'import numpy as np\nimport networkx as nx\nfrom shared import bincount_assigned\n\ncdef int UNMAPPED = -1\n\ndef get_votes(graph, int node, float[::] edge_weights, int num_partitions, int[::] partition):\n    seen = set()\n    cdef float[::] partition_votes = np.zeros(num_partitions, dtype=np.float32)\n\n    # find all neighbors from whole graph\n    node_neighbors = list(nx.all_neighbors(graph, node))\n    node_neighbors = [x for x in node_neighbors if x not in seen and not seen.add(x)]\n\n    # calculate votes based on neighbors placed in partitions\n    for n in node_neighbors:\n        if partition[n] != UNMAPPED:\n            partition_votes[partition[n]] += edge_weights[n]\n            \n    return partition_votes\n\ndef get_assignment(int node,\n                   float[::] node_weights,\n                   int num_partitions,\n                   int[::] partition,\n                   float[::] partition_votes,\n                   float alpha,\n                   int debug):\n\n    cdef int arg = 0\n    cdef int max_arg = 0\n    cdef float max_val = 0\n    cdef float val = 0\n    cdef int previous_assignment = 0\n\n    assert partition is not None, "Blank partition passed"\n\n    cdef float[::] partition_sizes = np.zeros(num_partitions, dtype=np.float32)\n    s = bincount_assigned(partition, num_partitions, weights=node_weights)\n    partition_sizes = np.fromiter(s, dtype=np.float32)\n    \n    if debug:\n        print("Assigning node {}".format(node))\n        print("\\tPn = Votes - Alpha x Size")\n\n    # Remember placement of node in the previous assignment\n    previous_assignment = partition[node]\n\n    max_arg = 0\n    max_val = partition_votes[0] - alpha * partition_sizes[0]\n    if debug:\n        print("\\tP{} = {} - {} x {} = {}".format(0,\n                                                 partition_votes[0],\n                                                 alpha,\n                                                 partition_sizes[0],\n                                                 max_val))\n\n    if previous_assignment == 0:\n        # We remove the node from its current partition before\n        # deciding to re-add it, so subtract alpha to give\n        # result of 1 lower partition size.\n        max_val += alpha\n\n    for arg in range(1, num_partitions):\n        val = partition_votes[arg] - alpha * partition_sizes[arg]\n\n        if debug:\n            print("\\tP{} = {} - {} x {} = {}".format(arg,\n                                                     partition_votes[arg],\n                                                     alpha,\n                                                     partition_sizes[arg],\n                                                     val))\n        if previous_assignment == arg:\n            # See comment above\n            val += alpha\n        if val > max_val:\n            max_arg = arg\n            max_val = val\n\n    if debug:\n        print("\\tassigned to P{}".format(max_arg))\n\n    return max_arg\n\ndef fennel_rework(graph, \n                  float[::] edge_weights,\n                  float[::] node_weights,\n                  int num_partitions,\n                  int[::] assignments,\n                  int[::] fixed,\n                  float alpha,\n                  int debug):\n\n    single_nodes = []\n    for n in range(0, graph.number_of_nodes()):\n\n        # Exclude single nodes, deal with these later\n        neighbors = list(nx.all_neighbors(graph, n))\n        if not neighbors:\n            single_nodes.append(n)\n            continue\n            \n        # Skip fixed nodes\n        if fixed[n] != UNMAPPED:\n            if debug:\n                print("Skipping node {}".format(n))\n            continue\n\n        partition_votes = get_votes(graph, n, edge_weights, num_partitions, assignments)\n        assignments[n] = get_assignment(n, node_weights, num_partitions, assignments, partition_votes, alpha, debug)\n\n    # Assign single nodes\n    for n in single_nodes:\n        if assignments[n] == UNMAPPED:\n            parts = bincount_assigned(assignments, num_partitions)\n            smallest = parts.index(min(parts))\n            assignments[n] = smallest\n\n    return np.asarray(assignments)')


# In[6]:

import shared
UNMAPPED = -1

# reset
assignments = np.repeat(np.int32(UNMAPPED), len(node_weights))
fixed = np.repeat(np.int32(UNMAPPED), len(node_weights))

print("PREDICTION MODEL")
print("----------------\n")
print("WASTE\t\tCUT RATIO\tMISMATCH")

if PREDICTION_MODEL:
    with open(PREDICTION_MODEL, "r") as inf:
        assignments = np.fromiter(inf.readlines(), dtype=np.int32)
    x = shared.score(assignments, edges)
    print("{0:.5f}\t\t{1:.10f}\t{2}".format(x[0], x[1], x[2]))

else:
    for i in range(num_iterations):
        alpha = prediction_model_alpha
        assignments = fennel_rework(G, edge_weights, node_weights, num_partitions, assignments, fixed, alpha, 0)

        x = shared.score(assignments, edges)
        print("{0:.5f}\t\t{1:.10f}\t{2}".format(x[0], x[1], x[2]))

print("\nAssignments:")
shared.fixed_width_print(assignments)

nodes_fixed = len([o for o in fixed if o == 1])
print("\nFixed: {}".format(nodes_fixed))

shared.print_partitions(assignments, num_partitions, node_weights)


# In[7]:

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

    edges = np.array(G.edges(), dtype=np.int32)
    edge_weights = np.array([x[2]['weight'] for x in G.edges(data=True)], dtype=np.float32)
    node_weights = np.array([x[1]['weight'] for x in G.nodes(data=True)], dtype=np.float32)

    print("\nAssignments:")
    shared.fixed_width_print(assignments)
    print("Last {} nodes are virtual nodes.".format(num_partitions))


# In[8]:

cut_off_value = int(prediction_model_cut_off * G.number_of_nodes())
if prediction_model_cut_off == 0:
    print("Discarding prediction model\n")
else:
    print("Assign first {} arrivals using prediction model, then discard\n".format(cut_off_value))

# fix arrivals
nodes_arrived = []
print("WASTE\t\tCUT RATIO\tMISMATCH")
for a in arrivals:
    nodes_fixed = len([o for o in fixed if o == 1])
    if nodes_fixed >= cut_off_value:
        break
    fixed[a] = 1
    nodes_arrived.append(a)

    # make a subgraph of all arrived nodes
    Gsub = G.subgraph(nodes_arrived)

    x = shared.score(assignments, Gsub.edges(), num_partitions)
    print("{0:.5f}\t\t{1:.10f}\t{2}".format(x[0], x[1], x[2]))

# remove nodes not fixed, ie. discard prediction model
for i in range(0, len(assignments)):
    if fixed[i] == -1:
        assignments[i] = -1

print("WASTE\t\tCUT RATIO\tMISMATCH")
x = shared.score(assignments, edges, num_partitions)
print("{0:.5f}\t\t{1:.10f}\t{2}".format(x[0], x[1], x[2]))

print("\nAssignments:")
shared.fixed_width_print(assignments)

nodes_fixed = len([o for o in fixed if o == 1])
print("\nFixed: {}".format(nodes_fixed))

shared.print_partitions(assignments, num_partitions, node_weights)


# In[9]:

if restream_batches == 1:
    print("One-shot assignment mode")
    print("------------------------\n")
else:
    print("Re-streaming in batches of {}".format(restream_batches))
    print("--------------------------------\n")

batch_arrived = []
print("WASTE\t\tCUT RATIO\tMISMATCH\tALPHA")
for a in arrivals:
    # check if node is already arrived
    if fixed[a] == 1:
        continue

    # one-shot assigment: assign each node as it arrives
    if restream_batches == 1:
        alpha = one_shot_alpha
        partition_votes = get_votes(G, a, edge_weights, num_partitions, assignments)
        assignments[a] = get_assignment(a, node_weights, num_partitions, assignments, partition_votes, alpha, 0)
        fixed[a] = 1
        nodes_arrived.append(a)

        # make a subgraph of all arrived nodes
        Gsub = G.subgraph(nodes_arrived)

        x = shared.score(assignments, Gsub.edges(), num_partitions)
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
            assignments[n] = get_assignment(n, node_weights, num_partitions, assignments, partition_votes, alpha, 0)
            fixed[n] = 1
            nodes_arrived.append(n)

        x = shared.score(assignments, Gsub.edges(), num_partitions)
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

shared.print_partitions(assignments, num_partitions, node_weights)


# In[10]:

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


# In[11]:

# Add partition attribute to nodes
for i in range(0, len(assignments)):
    G.add_nodes_from([i], partition=str(assignments[i]))

# Freeze Graph from further modification
G = nx.freeze(G)


# In[12]:

import os
import datetime
timestamp = datetime.datetime.now().strftime('%H%M%S')
data_filename,_ = os.path.splitext(os.path.basename(DATA_FILENAME))
data_filename += "-" + timestamp

# write to GML file
gml_filename = os.path.join(OUTPUT_DIRECTORY, data_filename + "-graph.gml")
print("Writing GML file: {}".format(gml_filename))
nx.write_gml(G, gml_filename)

# write assignments into a file with a single column
assignments_filename = os.path.join(OUTPUT_DIRECTORY, data_filename + "-assignments.txt")
print("Writing assignments: {}".format(assignments_filename))
with open(assignments_filename, "w") as outf:
    for i in range(0, len(assignments)):
        outf.write("{}\n".format(assignments[i]))

# write edge list into a file with, tab delimited
edges_filename = os.path.join(OUTPUT_DIRECTORY, data_filename + "-edges.txt")
print("Writing edge list: {}".format(edges_filename))
with open(edges_filename, "w") as outf:
    outf.write("{}\t{}\n".format(G.number_of_nodes(), G.number_of_edges()))
    for e in G.edges_iter():
        outf.write("{}\t{}\n".format(*e))
#nx.write_edgelist(G, edges_filename, delimiter='\t', data=False) # number of nodes/edges missing


# In[13]:

# original scoring algorithm
scoring = shared.score(assignments, G.edges(), num_partitions)

# edges cut and communication volume
edges_cut, steps = shared.base_metrics(G)

# MaxPerm
max_perm = 0.0
command = './bin/MaxPerm/MaxPerm < ' + edges_filename
os.system(command)
with open("./output.txt") as fp:
    for i, line in enumerate(fp):
        if "Network Permanence" in line:
            max_perm = line.split()[3]
            break
os.remove("./output.txt")

print("\nMetrics")
print("-------\n")

print("Edges cut: {}".format(edges_cut))
#print("Missmatch: {}".format(scoring[2]))
print("Waste: {}".format(scoring[0]))
print("Cut ratio: {}".format(scoring[1]))
print("Communication volume: {}".format(steps))
print("Network Permanence: {}".format(max_perm))

# write metrics to CSV
data = {
    "file": timestamp,
    "num_partitions": num_partitions,
    "num_iterations": num_iterations,
    "prediction_model_cut_off": prediction_model_cut_off,
    "one_shot_alpha": one_shot_alpha,
    "restream_batches": restream_batches,
    "use_virtual_nodes": use_virtual_nodes,
    "virtual_node_weight": virtual_node_weight,
    "virtual_edge_weight": virtual_edge_weight,
    "edges_cut": edges_cut,
    "waste": scoring[0],
    "cut_ratio": scoring[1],
    "communication_volume": steps,
    "network_permanence": max_perm
}
fieldnames = [
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
    "network_permanence"
]
    
import csv
metrics_filename = os.path.join(OUTPUT_DIRECTORY, "metrics.csv")
if not os.path.exists(metrics_filename):
    with open(metrics_filename, "w", newline='') as outf:
        csv_writer = csv.DictWriter(outf, fieldnames=fieldnames)
        csv_writer.writeheader()
with open(metrics_filename, "a", newline='') as outf:
    csv_writer = csv.DictWriter(outf, fieldnames=fieldnames)
    csv_writer.writerow(data)


# In[ ]:



