
# Cleaning the data

import os
import numpy as np
import gzip

def row_generator(data_path):
    """This will generate all the edges in the graph."""
    edges = []
    with gzip.open(data_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            else:
                (left_node, right_node) = line[:-1].split()
                edges.append((int(left_node), int(right_node)))
    num_edges = len(edges)
    # XXX: max() might be a mistake here, use len(set()) instead?
    num_nodes = max([x[0] for x in edges] + [x[1] for x in edges]) + 1
    return edges, num_edges, num_nodes

def to_undirected(edge_iterable, num_edges, num_nodes, shuffle=True):
    """Takes an iterable of edges and produces the list of edges for the undirected graph.

    > to_undirected([[0,1],[1,2],[2,10]], 3, 11)
    array([[ 0,  1],
       [ 1,  0],
       [ 1,  2],
       [ 2,  1],
       [ 2, 10],
       [10,  2]])
    """
    # need int64 to do gross bithacks
    as_array = np.zeros((num_edges, 2), dtype=np.int64)
    for (i, (n_0, n_1)) in enumerate(edge_iterable):
            as_array[i,0] = n_0
            as_array[i,1] = n_1
    # The graph is directed, but we want to make it undirected,
    # which means we will duplicate some rows.

    left_nodes = as_array[:,0]
    right_nodes = as_array[:,1]

    if shuffle:
        the_shuffle = np.arange(num_nodes)
        np.random.shuffle(the_shuffle)
        left_nodes = the_shuffle.take(left_nodes)
        right_nodes = the_shuffle.take(right_nodes)


    # numpy.unique will not unique whole rows, so this little bit-hacking
    # is a quick way to get unique rows after making a flipped copy of
    # each edge.
    max_bits = int(np.ceil(np.log2(num_nodes + 1)))

    encoded_edges_forward = np.left_shift(left_nodes, max_bits) | right_nodes

    # Flip the columns and do it again:
    encoded_edges_reverse = np.left_shift(right_nodes, max_bits) | left_nodes

    unique_encoded_edges = np.unique(np.hstack((encoded_edges_forward, encoded_edges_reverse)))

    left_node_decoded = np.right_shift(unique_encoded_edges, max_bits)

    # Mask out the high order bits
    right_node_decoded = (2 ** (max_bits) - 1) & unique_encoded_edges

    undirected_edges = np.vstack((left_node_decoded, right_node_decoded)).T.astype(np.int32)

    # ascontiguousarray so that it's c-contiguous for cython code below
    return np.ascontiguousarray(undirected_edges)


def get_clean_data(data_path, shuffle=True, save_readable=False):
    data_dir = os.path.dirname(data_path)
    file_name, _ = os.path.splitext(os.path.basename(data_path))

    if shuffle:
        name = os.path.join(data_dir, file_name + '-cleaned-shuffled.npy')
        name_readable = os.path.join(data_dir, file_name + '-cleaned-shuffled.txt')
    else:
        name = os.path.join(data_dir, file_name + '-cleaned.npy')
        name_readable = os.path.join(data_dir, file_name + '-cleaned.txt')

    if False and os.path.exists(name):
        print('Loading from file {}'.format(name))
        return np.load(name)
    else:
        print('Parsing from zip. Will write to file {}'.format(name), flush=True)

        # Lets get the edges into one big array
        edges, num_edges, num_nodes = row_generator(data_path)
        edges = to_undirected(edges, num_edges, num_nodes, shuffle=shuffle)
        print('ORIGINAL DIST: {} MIN: {} MAX: {}'.format(np.abs(edges[:,0] - edges[:,1]).mean(), edges.min(), edges.max()))
        np.save(name, edges)

        if save_readable:
            with open(name_readable, 'w') as r:
                for e in edges:
                    r.write("{} {}\n".format(e[0], e[1]))

        return edges, num_edges, num_nodes


def bincount_assigned(a, n):
    parts = [0] * n
    for i in range(0, len(a)):
        if a[i] >= 0:
            parts[a[i]] += 1
    return parts

def score(assignment, edges, n=None):
    """Compute the score given an assignment of vertices.

    N nodes are assigned to clusters 0 to K-1.

    assignment: Vector where N[i] is the cluster node i is assigned to.
    edges: The edges in the graph, assumed to have one in each direction

    Returns: (total wasted bin space, ratio of edges cut)
    """
    if n:
        balance = np.array(bincount_assigned(assignment, n)) / len(assignment)
    else:
        balance = np.bincount(assignment) / len(assignment)
    waste = (np.max(balance) - balance).sum()

    left_edge_assignment = assignment.take(edges[:,0])
    right_edge_assignment = assignment.take(edges[:,1])
    mismatch = (left_edge_assignment != right_edge_assignment).sum()
    cut_ratio = mismatch / len(edges)

    return (waste, cut_ratio, mismatch)


def print_partitions(assignments, num_partitions, node_weights):

    if -1 not in assignments:
        print("\nPartitions - nodes (weight):")
        partition_size_nodes = np.bincount(assignments, minlength=num_partitions).astype(np.float32)
        partition_size_weights = np.bincount(assignments, weights=node_weights, minlength=num_partitions).astype(np.float32)
        for p in range(0, num_partitions):
            print("P{}: {} ({})".format(p, partition_size_nodes[p], partition_size_weights[p]))

    else:
        print("\nPartitions - nodes:")
        parts = [0] * num_partitions
        for i in range(0, len(assignments)):
            if assignments[i] >= 0:
                parts[assignments[i]] += 1
        for p in range(0, len(parts)):
            print("P{}: {}".format(p, parts[p]))


def fixed_width_print(arr):
    print("[", end='')
    for x in range(0, len(arr)):
        if arr[x] >= 0:
            print(" ", end='')

        print("{}".format(arr[x]), end='')

        if x != len(arr)-1:
            print(" ", end='')
    print("]")

def line_print(assignments):
    for i in range(0, len(assignments)):
        for b in range(i, len(assignments)):
            if assignments[b] != -1:
                break
        if b != len(assignments)-1:
            print("{} ".format(assignments[i]), end='')
    print()

# write to file
def write_to_file(filename, assignments):
    with open(filename, "w") as f:
        j = 0
        for a in assignments:
            f.write("{} {}\n".format(j,a))
            j += 1

