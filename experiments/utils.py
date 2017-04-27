import os
import datetime

def getOutFileName(algorithm, partition, correctedness, cutoff, networkID, sliding, virtualNodes, gam):
    '''
    Creates a filename for an experiment, based on the experiment's parameters
    '''
    fn = algorithm + "_p" + str(partition) + "_c" + str(correctedness) + "_cutoff" + str(int(cutoff * 100))
    fn += "_sw" + str(int(sliding)) + "_vn" + str(int(virtualNodes)) + "_gam" + str(int(gam)) + "_" + str(networkID)
    return fn + ".txt"

def getConfig(parametrized_config, algorithm, partition, correctedness, cutoff, networkID, sliding, virtualNodes, gam):
    '''
        Takes the parametrized config variable, replaces and updates the correct parameters
    '''
    newConfig = parametrized_config.copy()
    for confKey in list(newConfig.keys()):
        changed = False
        conf = newConfig[confKey]
        try:
            if "#networkID#" in str(conf):
                conf = conf.replace("#networkID#", str(networkID))
                changed = True
            if "#correctedness#" in str(conf):
                conf = conf.replace("#correctedness#", str(correctedness))
                changed = True
        except Excpetion as err:
            pass
        if changed:
            newConfig[confKey] = conf
    newConfig["PREDICTION_MODEL_ALGORITHM"] = algorithm
    newConfig["PARTITIONER_ALGORITHM"] = algorithm
    newConfig['num_partitions'] = partition
    newConfig["prediction_model_cut_off"] = cutoff
    newConfig["sliding_window"] = sliding
    newConfig["use_virtual_nodes"] = virtualNodes
    newConfig["alter_node_weight_to_gam_prediction"] = gam
    return newConfig

def experimentParentDir(parametrized_config):
    return os.path.join(parametrized_config["OUTPUT_DIRECTORY"], datetime.datetime.now().strftime('%y_%m_%d').replace("/", ""))

def experimentDir(parametrized_config):
    return os.path.join(experimentParentDir(parametrized_config), datetime.datetime.now().strftime('%H_%M_%S'))

def purgeEmptyDir(directory):
    for dirpath, dirnames, files in os.walk(directory):
        if not files:
            os.rmdir(directory)

def cutoffValueFunc(val):
    return val * 0.05

def times10(val):
    return val * 10

def fillRange(minN, maxN, valueFunc = None):
    values = []
    for p in range(minN, maxN):
        if valueFunc is None:
            values.append(p)
        else:
            values.append(valueFunc(p))
    return values
