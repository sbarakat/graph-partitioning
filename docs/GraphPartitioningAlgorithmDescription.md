# Graph Partitioning

## Load Network

Graph is loaded via metis and initial number of nodes is stored

Prediction model alpha is computed at this stage

Arrival order is in range 0 - number of nodes

If simulated arrival file is not present, then all nodes are assumed to arrive.

If no prediction list file, then displacement weights = 1 for each node

if Graph modification functions, edge and node weights are stored

Copy of original graph is made


## Init Partitioner

If FENNEL, then Fennel Partitioner is loaded, with prediction_model_alpha passed.

If SCOTCH, then ScotchPartitioner is loaded with use virtual nodes passed as parameter

If PATOH, then PathohPrtitioner is loaded, with patoh iterations and hyperedge expansion parameters. Normal edge expansion is disabled since we use hyperedge expansion

## Prediction Model

SCOTCH is set to use a 'quality' partition strategy which optimises for this property.

If set, Graph Prediction Weights are applied: Graph node weights are set to predicted displacement weights

If prediction model file present, the assignments are loaded from disk

If Graph modification functions enabled, Graph edge expansion is performed (N.B. the edge expansion function checks if it is enabled or disabled, see for patoh)

Prediction model assignments are computed and stored in assigments.

Metrics are computed

If virtual nodes are enabled, these are now initialised

If prediction model weights are applied, they are now removed from the graph since the prediction model has been computed

### Virtual Node initialisation
---
One virtual node for each partition is created

Each node in the graph is then connected to the virtual node in its same partition

The assignments and fixed arrays are extended to include the virtual nodes

Virtual nodes and edges are then added to the graph object

## Assign Cutoff

The cutoff point is computed as: the % cutoff value * the number of people who should arrive. So that a specific proportion of people who arrive is assigned automatically, based on their prediction model partition.

For each node in arrival order, we perform the following:

1. check if the node is in the simulated arrivals list
2. if graph modification function enabled and alter arrived node weight to 100 is set, then that node's weight is set to 100
3. we count number of fixed nodes, if it is less than the cutoff value, we fix the current node and add it to the nodes arrived list

We then set the assignments value to -1 for all nodes that are not fixed at the moment and generate the graph metrics for the subgraph of nodes that have arrived


## Batch Arrival

SCOTCH partition strategy is set to balanced rather than quality, as this ensures that the resulting partitions have a balanced equal total weight

For each node in arrival node we perform the following:

1. ignore it if it is an already fixed node and move on to the next node
2. if graph modification functions enabled, then we: a) remove the node if is not a node that is in the simulated arrivals list b) set the node weight to 100 if alter arrivde node weight to 100 is set, for those nodes that are expected to arrive
3. We skip nodes that are not in the simulated arrivals list (this is already performed in part 2 if graph modfication functions is enabled, so here we don't remove the node but just skip this node)
4. we store the nodes that make it this far into the current batch list
5. if the current batch matches the required batch size, then we call **process_batch** and partition it with the other nodes that have already arrived
6. If we are not in sliding window mode, then the batch is reset to empty at this point

At the end of the loop, we process any remaining nodes that arrive into a final batch, which may be smaller than the required batch size. This is performed with ```assign_all=True```

After the batch arrival, we remove all the nodes that are not fixed from the assignments array, setting them to -1

N.B. process batch function computes the batch metrics which are stored into a list for each iteration.

### Process batch
---

If graph modification functions are enabled:


1. Compute the total number of people arrived
2. If the total arrived is < gam_k_value, k is the total number of people arrived, otherwise, the gam_k_value
3. The gam_prediction weights are computed and are then assigned as node weights to the graph
4. We compute edge expansion on the graph, if enabled (not patoh)

A subgraph is generated that contains only the nodes and edges for the people who have arrived

The number of edges arrived is then computed

For Fennel, we recompute the current alpha value

If we alter the node weights to gam prediction, then the assignments are computed on the whole graph, not the subgraph, otherwise, the partitioning for the assignments is computed on the subgraph only.

In sliding window mode, the first node is removed from the current batch and is fixed and assigned to the list of arrived nodes. The other nodes in the batch are ignored.

In non sliding window mode, all the nodes in the batch are fixed and added to the nodes arrived list.

The scores are computed for the subgraph and returned to be stored in the batch arrival function.

## Metrics computation functions

### ```utilities.score(graph, assignments, num partitions)```

This computes the WASTE and Cut RATIO metrics

TODO details

### ```utils.base_metrics(graph, assignments)```

Computes the edges_cut, total communication volume and the list of edges that have been cut.

### ```utils.modularity_wavg(graph, assignments, num partitions)```

Computes the modularity.

### ```utils.loneliness_score_wavg(graph, loneliness_score_param, assignments, num partitions)```

Computes the loneliness score.

### NMI, Total cut weight, fscore

Using the list of cut edges, we take the weight for that edge and add it to the total value of each cut edge.

```utils.fscore2(prediction_model_assignments, assignments, num_partition)``` computes the fscore between the assignments and prediction model assignments. It also performs node-relabelling and returns the improvement in fscore with the relabeling of nodes.
