import os
import csv
import gzip
import shutil
import tempfile
import itertools
import subprocess
import community
import numpy as np
import networkx as nx

BIN_DIRECTORY = os.path.join(os.path.dirname(__file__), "..", "bin")

def read_metis(DATA_FILENAME):

    G = nx.Graph()

    # add node weights from METIS file
    with open(DATA_FILENAME, "r") as metis:

        n = 0
        first_line = None
        has_edge_weights = False
        has_node_weights = False
        for i, line in enumerate(metis):
            if line[0] == '%':
                # ignore comments
                continue

            if not first_line:
                # read meta data from first line
                first_line = line.split()
                m_nodes = int(first_line[0])
                m_edges = int(first_line[1])
                if len(first_line) > 2:
                    # FMT has the following meanings:
                    #  0  the graph has no weights (in this case, you can omit FMT)
                    #  1  the graph has edge weights
                    # 10  the graph has node weights
                    # 11  the graph has both edge and node weights
                    file_format = first_line[2]
                    if int(file_format) == 0:
                        pass
                    elif int(file_format) == 1:
                        has_edge_weights = True
                    elif int(file_format) == 10:
                        has_node_weights = True
                    elif int(file_format) == 11:
                        has_edge_weights = True
                        has_node_weights = True
                    else:
                        assert False, "File format not supported"
                continue

            # METIS starts node count from 1, here we start from 0 by
            # subtracting 1 in the edge list and incrementing 'n' after
            # processing the line.
            if line.strip():
                e = line.split()
                if has_edge_weights and has_node_weights:
                    if len(e) > 2:
                        # create weighted edge list:
                        #  [(1, 2, {'weight':'2'}), (1, 3, {'weight':'8'})]
                        edges_split = list(zip(*[iter(e[1:])] * 2))
                        edge_list = [(n, int(v[0]) - 1, {'weight': int(v[1])}) for v in edges_split]

                        G.add_edges_from(edge_list)
                        G.node[n]['weight'] = int(e[0])
                    else:
                        # no edges
                        G.add_nodes_from([n], weight=int(e[0]))

                elif has_edge_weights and not has_node_weights:
                    pass
                elif not has_edge_weights and has_node_weights:
                    pass
                else:
                    edge_list = [(n, int(v) - 1, {'weight':1.0}) for v in e]
                    G.add_edges_from(edge_list)
                    G.node[n]['weight'] = 1.0
            else:
                # blank line indicates no node weight
                G.add_nodes_from([n], weight=1.0)
            n += 1

    # sanity check
    assert (m_nodes == G.number_of_nodes()), "Expected {} nodes, networkx graph contains {} nodes".format(m_nodes, G.number_of_nodes())
    assert (m_edges == G.number_of_edges()), "Expected {} edges, networkx graph contains {} edges".format(m_edges, G.number_of_edges())

    return G


def bincount_assigned(graph, assignments, num_partitions):
    parts = [0] * num_partitions
    for n in graph.nodes_iter(data=True):
        node = n[0]
        if 'weight' in n[1]:
            weight = n[1]['weight']
        else:
            weight = 1
        if assignments[node] >= 0:
            parts[assignments[node]] += weight

    return parts


rpy2_loaded = False
base = None
utils = None
mgcv = None
def gam_predict(location_csv, prediction_file, num_arrived, k_value):

    if not rpy2_loaded:
        from rpy2.robjects import Formula
        from rpy2.robjects.packages import importr
        base = importr('base')
        utils = importr('utils')
        mgcv = importr('mgcv')
        location_filename = os.path.basename(location_csv)
        prediction_filename = os.path.basename(prediction_file)

    # Setup
    base.setwd(os.path.dirname(location_csv))
    loc = utils.read_csv(location_filename, header=False, nrows=num_arrived)
    pred = utils.read_csv(prediction_filename, header=False, nrows=num_arrived)
    pop = base.cbind(pred, loc)
    pop.colnames = ["shelter","x","y"]

    # GAM
    formula = Formula('shelter~s(x,y,k={})'.format(k_value))
    m = mgcv.gam(formula, family="binomial", method="REML", data=pop)

    # Predict for everyone
    loc = utils.read_csv(location_filename, header=False)
    pred = utils.read_csv(prediction_filename, header=False)
    newd = base.cbind(pred, loc)
    newd.colnames = ["shelter","x","y"]
    result = mgcv.predict_gam(m, newd, type="response", se_fit=False)

    return list(result)


def score(graph, assignment, num_partitions=None):
    """Compute the score given an assignment of vertices.

    N nodes are assigned to clusters 0 to K-1.

    assignment: Vector where N[i] is the cluster node i is assigned to.
    edges: The edges in the graph, assumed to have one in each direction

    Returns: (total wasted bin space, ratio of edges cut)
    """
    if num_partitions:
        balance = np.array(bincount_assigned(graph, assignment, num_partitions)) / len(assignment)
    else:
        balance = np.bincount(assignment) / len(assignment)
    waste = (np.max(balance) - balance).sum()

    left_edge_assignment = assignment.take([x[0] for x in graph.edges()]) #edges[:,0])
    right_edge_assignment = assignment.take([x[1] for x in graph.edges()]) #edges[:,1])
    mismatch = (left_edge_assignment != right_edge_assignment).sum()
    if graph.number_of_edges() > 0:
        cut_ratio = mismatch / graph.number_of_edges()
    else:
        cut_ratio = 0.0

    return (waste, cut_ratio, mismatch)


def base_metrics(G, assignments=None):
    """
    This algorithm calculates the number of edges cut and scores the communication steps. It gets
    passed a networkx graph with a 'partition' attribute defining the partition of the node.

    Communication steps described on slide 11:
    https://www.cs.fsu.edu/~engelen/courses/HPC-adv/GraphPartitioning.pdf
    """
    steps = 0
    edges_cut = 0
    seen = []
    for n in G.nodes_iter():
        partition_seen = []
        for e in G.edges_iter(n):
            left = e[0]
            right = e[1]
            if assignments is None:
                left_partition = G.node[left]['partition']
                right_partition = G.node[right]['partition']
            else:
                left_partition = assignments[left]
                right_partition = assignments[right]

            if left_partition == right_partition:
                # right node within same partition, skip
                continue

            if (n,right) not in seen:
                # dealing with undirected graphs
                seen.append((n,right))
                seen.append((right,n))

                if left_partition != right_partition:
                    # right node in different partition
                    edges_cut += 1

            if left_partition != right_partition and right_partition not in partition_seen:
                steps += 1
                partition_seen.append(right_partition)

    mod = None
    if not assignments is None:
        part = dict(zip(G.nodes(), assignments))
        mod = community.modularity(part, G)

    return (edges_cut, steps, mod)

def modularity(G, best_partition=False):
    if best_partition:
        part = community.best_partition(G)
    else:
        part = dict([(n[0], int(n[1]['partition'])) for n in G.nodes(data=True)])
    mod = community.modularity(part, G)
    return mod


def loneliness_score(G, loneliness_score_param):
    total = 0
    count = 0
    for n in G.nodes():
        node_edges = len(G[n])
        score = 1 - ((1 / (node_edges + 1)**loneliness_score_param))
        total += score
        count += 1

    # average for partition
    return total / count

def complete_loneliness_score(G, loneliness_score_param, assignments, num_partitions):
    """
    Return a weighted average across all partitions for loneliness score
    """
    p = get_partition_population(G, assignments, num_partitions)
    partition_population = [p[x][0] for x in p]
    partition_score = list(range(0, num_partitions))

    for p in range(0, num_partitions):
        nodes = [i for i,x in enumerate(assignments) if x == p]
        Gsub = G.subgraph(nodes)
        partition_score[p] = loneliness_score(Gsub, loneliness_score_param)

    return np.average(partition_score, weights=partition_population)


def run_max_perm(edges_maxperm_filename):
    max_perm = 0.0
    temp_dir = tempfile.mkdtemp()
    with open(edges_maxperm_filename, "r") as edge_file:
        args = [os.path.join(BIN_DIRECTORY, "MaxPerm", "MaxPerm")]
        retval = subprocess.call(
            args, cwd=temp_dir, stdin=edge_file,
            stderr=subprocess.STDOUT)
    with open(os.path.join(temp_dir, "output.txt"), "r") as fp:
        for i, line in enumerate(fp):
            if "Network Permanence" in line:
                max_perm = line.split()[3]
                break
    shutil.rmtree(temp_dir)
    return max_perm


def run_community_metrics(output_path, data_filename, edges_oslom_filename):
    """
    Community Quality metrics
    Use OSLOM to find clusters in edgelist, then run ComQualityMetric to get metrics.

    http://www.oslom.org/
    https://github.com/chenmingming/ComQualityMetric
    """
    temp_dir = tempfile.mkdtemp()
    oslom_bin = os.path.join(BIN_DIRECTORY, "OSLOM2", "oslom_dir")
    oslom_log = os.path.join(output_path, 'oslom', data_filename + "-oslom.log")
    oslom_modules = os.path.join(output_path, 'oslom', data_filename + "-oslom-tp.txt")
    args = [oslom_bin, "-f", edges_oslom_filename, "-w", "-r", "10", "-hr", "50"]
    with open(oslom_log, "w") as logwriter:
        retval = subprocess.call(
            args, cwd=temp_dir,
            stdout=logwriter, stderr=subprocess.STDOUT)
    shutil.copy(os.path.join(temp_dir, "tp"), oslom_modules)
    shutil.rmtree(temp_dir)

    com_qual_path = os.path.join(BIN_DIRECTORY, "ComQualityMetric")
    com_qual_log = os.path.join(output_path, 'oslom', data_filename + "-CommunityQuality.log")
    args = ["java", "OverlappingCommunityQuality", "-weighted", edges_oslom_filename, oslom_modules]
    with open(com_qual_log, "w") as logwriter:
        retval = subprocess.call(
            args, cwd=com_qual_path,
            stdout=logwriter, stderr=subprocess.STDOUT)

    if retval != 0:
        with open(com_qual_log, "r") as log:
            e = log.read().replace('\n', '')
        raise Exception(e)

    with open(com_qual_log, "r") as fp:
        metrics = {}
        for line in fp:
            if ' = ' in line:
                m = [p.strip() for p in line.split(',')]
                metrics.update(dict(map(lambda y:y.split(' = '), m)))

    return metrics

def get_partition_population(graph, assignments, num_partitions):
    population = {}
    if -1 not in assignments:
        nodes = np.bincount(assignments, minlength=num_partitions).astype(np.float32)
        weights = bincount_assigned(graph, assignments, num_partitions)

    else:
        nodes = [0] * num_partitions
        weights = [0] * num_partitions
        for i in range(0, len(assignments)):
            if assignments[i] >= 0:
                nodes[assignments[i]] += 1

    for p in range(0, num_partitions):
        population[p] = (nodes[p], weights[p])

    return population

def print_partitions(graph, assignments, num_partitions):
    population = get_partition_population(graph, assignments, num_partitions)

    print("\nPartitions - nodes (weight):")
    for p in population:
        print("P{}: {} ({})".format(p, population[p][0], population[p][1]))

def fixed_width_print(arr):
    print("[", end='')
    for x in range(0, len(arr)):
        if arr[x] >= 0:
            print(" ", end='')

        print("{}".format(arr[x]), end='')

        if x != len(arr)-1:
            print(" ", end='')
    print("]")


def write_graph_files(output_path, data_filename, G, quiet=False, relabel_nodes=False):

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(os.path.join(output_path, 'oslom')):
        os.makedirs(os.path.join(output_path, 'oslom'))
    if not os.path.exists(os.path.join(output_path, 'maxperm')):
        os.makedirs(os.path.join(output_path, 'maxperm'))
    if not os.path.exists(os.path.join(output_path, 'graphs')):
        os.makedirs(os.path.join(output_path, 'graphs'))

    # write to GML file
    gml_filename = os.path.join(output_path, 'graphs', data_filename + "-graph.gml")
    nx.write_gml(G, gml_filename)

    # write assignments into a file with a single column
    assignments_filename = os.path.join(output_path, 'graphs', data_filename + "-assignments.txt")
    with open(assignments_filename, "w") as outf:
        for n in G.nodes_iter(data=True):
            outf.write("{}\n".format(n[1]["partition"]))

    # write edge list in a format for OSLOM, tab delimited
    edges_oslom_filename = os.path.join(output_path, 'oslom', data_filename + "-edges-oslom.txt")
    with open(edges_oslom_filename, "w") as outf:
        for e in G.edges_iter(data=True):
            outf.write("{}\t{}\t{}\n".format(e[0], e[1], e[2]["weight"]))

    # MaxPerm requires nodes in sequential order
    if relabel_nodes:
        mapping = dict(zip(G.nodes(), range(0, len(G.nodes()))))
        nx.relabel_nodes(G, mapping, copy=False)

    # write edge list in a format for MaxPerm, tab delimited
    edges_maxperm_filename = os.path.join(output_path, 'maxperm', data_filename + "-edges-maxperm.txt")
    with open(edges_maxperm_filename, "w") as outf:
        outf.write("{}\t{}\n".format(G.number_of_nodes(), G.number_of_edges()))
        for e in sorted(G.edges_iter()):
            outf.write("{}\t{}\n".format(*e))

    if not quiet:
        print("Writing GML file: {}".format(gml_filename))
        print("Writing assignments: {}".format(assignments_filename))
        print("Writing edge list (for MaxPerm): {}".format(edges_maxperm_filename))
        print("Writing edge list (for OSLOM): {}".format(edges_oslom_filename))

    return (edges_maxperm_filename, edges_oslom_filename)

def write_metrics_csv(filename, fields, metrics):
    if not os.path.exists(filename):
        with open(filename, "w", newline='') as outf:
            csv_writer = csv.DictWriter(outf, fieldnames=fields)
            csv_writer.writeheader()
    with open(filename, "a", newline='') as outf:
        csv_writer = csv.DictWriter(outf, fieldnames=fields)
        csv_writer.writerow(metrics)

