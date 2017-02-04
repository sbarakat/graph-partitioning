import os
import shared
import argparse
import networkx as nx
from graph_partitioning import GraphPartitioning


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


config = {

    "DATA_FILENAME": args.data_file,
    "OUTPUT_DIRECTORY": args.output_dir,

    # Read input file for prediction model, if not provided a prediction
    # model is made using FENNE
    "PREDICTION_MODEL": "",

    # File containing simulated arrivals. This is used in simulating nodes
    # arriving at the shelter. Nodes represented by line number; value of
    # 1 represents a node as arrived; value of 0 represents the node as not
    # arrived or needing a shelter.
    "SIMULATED_ARRIVAL_FILE": os.path.join(pwd, "data", "simulated_arrival.txt"),

    # File containing the geographic location of each node.
    "POPULATION_LOCATION_FILE": os.path.join(pwd, "data", "population_location.csv"),

    # Number of shelters
    "num_partitions": args.num_partitions,

    # The number of iterations when making prediction model
    "num_iterations": args.num_iterations,

    # Percentage of prediction model to use before discarding
    # When set to 0, prediction model is discarded, useful for one-shot
    "prediction_model_cut_off": args.prediction_model_cut_off,

    # Alpha value used in one-shot (when restream_batches set to 1)
    "one_shot_alpha": args.one_shot_alpha,

    # Number of arrivals to batch before recalculating alpha and restreaming.
    # When set to 1, one-shot is used with alpha value from above
    "restream_batches": args.restream_batches,

    # When the batch size is reached: if set to True, each node is assigned
    # individually as first in first out. If set to False, the entire batch
    # is processed and empty before working on the next batch.
    "sliding_window": False,

    # Create virtual nodes based on prediction model
    "use_virtual_nodes": args.use_virtual_nodes,

    # Virtual nodes: edge weight
    "virtual_edge_weight": 1.0,


    ####
    # GRAPH MODIFICATION FUNCTIONS

    # Also enables the edge calculation function.
    "graph_modification_functions": True,

    # If set, the node weight is set to 100 if the node arrives at the shelter,
    # otherwise the node is removed from the graph.
    "alter_arrived_node_weight_to_100": True,

    # Uses generalized additive models from R to generate prediction of nodes not
    # arrived. This sets the node weight on unarrived nodes the the prediction
    # given by a GAM.
    # Needs POPULATION_LOCATION_FILE to be set.
    "alter_node_weight_to_gam_prediction": True,

    # The value of 'k' used in the GAM will be the number of nodes arrived until
    # it reaches this max value.
    "gam_k_value": 100,

    # Alter the edge weight for nodes that haven't arrived. This is a way to
    # de-emphasise the prediction model for the unknown nodes.
    "prediction_model_emphasis": 1.0,
}

gp = GraphPartitioning(config)

# Optional: shuffle the order of nodes arriving
# Arrival order should not be shuffled if using GAM to alter node weights
#random.shuffle(gp.arrival_order)

gp.load_network()
gp.prediction_model()
gp.assign_cut_off()
gp.batch_arrival()
gp.get_metrics()

