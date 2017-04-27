import graph_partitioning.utils as utils

class OutFile:
    def __init__(self, filePath):
        self.contents = []
        self.filePath = filePath

    def write(self, moreContent):
        self.contents.append(moreContent)

    def save(self):
        with open(self.filePath, 'w+') as f:
            for content in self.contents:
                f.write(content)

    def load(self):
        predictionModelData = []
        batchModelData = []
        with open(self.filePath, 'r') as f:
            metricsFoundCount = 0
            metricsEndCount = 0

            for line in f:
                line = line.strip()

                if 'ENDSTATS' in line:
                    metricsEndCount += 1

                if metricsEndCount < metricsFoundCount:
                    if 'WASTE' in line:
                        continue
                    # extract metrics
                    if metricsFoundCount == 1:
                        predictionModelData = self._convertCSVLineToFloat(line)
                    elif metricsFoundCount == 3:
                        batchModelData.append(self._convertCSVLineToFloat(line))

                if 'METRICS' in line:
                    metricsFoundCount += 1

        return (predictionModelData, batchModelData)


    def _load(self):
        predictionModelData = []
        batchModelData = []
        with open(self.filePath, 'r') as f:
            count = 1
            for line in f:
                line = line.strip()

                if(count == 45):
                    predictionModelData = self._convertCSVLineToFloat(line)

                if (count >= 121):
                    if 'ENDSTATS' in line:
                        break
                    batchModelData.append(self._convertCSVLineToFloat(line))
                count += 1
        return (predictionModelData, batchModelData)

    def _convertCSVLineToFloat(self, line):
        arr = []
        for part in line.split(", "):
            arr.append(float(part))
        return arr

def writeConfig(outFile, config, numPartitions, networkID):
    outFile.write("PREDICTION_MODEL_ALGORITHM = " + str(config["PREDICTION_MODEL_ALGORITHM"]) + "\n")
    outFile.write("PARTITIONER_ALGORITHM = " + str(config["PARTITIONER_ALGORITHM"]) + "\n")
    outFile.write("partitions = " + str(numPartitions) + "\n")
    outFile.write("networkID = " + str(networkID) + "\n")
    outFile.write("prediction_model_cut_off = " + str(config["prediction_model_cut_off"]) + "\n")
    outFile.write("restream_batches = " + str(config["restream_batches"]) + "\n")
    outFile.write("sliding_window = " + str(config["sliding_window"]) + "\n")
    outFile.write("virtual_nodes = " + str(config["use_virtual_nodes"]) + "\n")
    outFile.write("gam = " + str(config["alter_node_weight_to_gam_prediction"]) + "\n")

def writeArray(outFile, arrayName, array, valuesPerLine = 50):
    outFile.write("\nSTARTARRAY-" + arrayName + "\n[")

    line = ""
    count = 0
    isFirst = True
    for value in array:
        if(len(line)):
            # add comma to separate from previous
            line += ", "
        # add new line if we've reached valuesPerLine
        if(count == valuesPerLine):
            count = 0;
            line += "\n"
        line += str(value)
        count += 1
    if len(line) > 0:
        outFile.write(line)
    outFile.write("]\nENDARRAY\n")

def writePartitionStats(outFile, section, gp, m):
    #print("writePartitionStats", outFile.contents)
    writeArray(outFile, section, gp.assignments)

    population = utils.get_partition_population(gp.G, gp.assignments, gp.num_partitions)

    outFile.write("\nSTARTSTATS FOR " + section + "\n")
    outFile.write("Partitions - nodes (weight)\n")
    for p in population:
        outFile.write("P{}: {} ({})\n".format(p, population[p][0], population[p][1]))

    # print dataframe stats now
    outFile.write("\nMETRICS\n")
    outFile.write("WASTE, CUT RATIO, EDGES CUT, TOTAL COMM VOLUME, MODULARITY, LONELINESS, NETWORK PERMANENCE, NORM. MUTUAL INFO\n")
    for row in m:
        line = ""
        for value in row:
            if len(line) > 0:
                line += ", "
            line += str(value)
        outFile.write(line + "\n")
    outFile.write("ENDSTATS\n")
