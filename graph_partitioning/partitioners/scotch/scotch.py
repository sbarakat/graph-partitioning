
import graph_partitioning.partitioners.scotch.scotch_data as sd
import graph_partitioning.partitioners.scotch.lib_scotch as slib

def isScotchData(data):
    return isinstance(data, sd.ScotchData)

def isValidScotchData(data):
    if isScotchData(data) == False:
        return False

    return data.isValid()

class Scotch:
    def __init__(self, scotchLibPath = None, numPartitions = 10, kbalval = 0.01, strategyFlag = 1, strategyOptions = ''):
        self.scotchLib = slib.LibScotch(libraryPath=scotchLibPath)
        self.scotchLib.load()

        self.scotchData = None

        # Optional Parameters
        self.numPartitions = numPartitions
        self.kbalval = kbalval
        self.strategyFlag = strategyFlag
        self.strategyOptions = strategyOptions

    def initialize(self, scotchArrayData, verbose=True, skipGraphValidStep=False):
        if(verbose):
            print("Intializing Architecture for GraphMap")

        ok = self.initArchitecture()

        if(verbose):
            print("   Architecture =", ok)

        if(verbose):
            print("Intializing Strategy for GraphMap")

        ok = self.initStrategy()

        if(verbose):
            print("   Strategy =", ok)

        if(verbose):
            print("Loading Graph for GraphMap")

        ok = self.loadGraph(scotchArrayData, skipGraphValidStep=skipGraphValidStep)

        if(verbose):
            print("   Graph =", ok)

        return ok


    def delObjects(self):
        if self.scotchLib.isLoaded():
            self.scotchLib.deleteSCOTCHGraph()
            self.scotchLib.deleteSCOTCHStrat()
            self.scotchLib.deleteSCOTCHArch()
        if self.scotchData is not None:
            # clear arrays
            self.scotchData.clearData()


    def initArchitecture(self):
        if self.scotchLib.isLoaded():
            ok = self.scotchLib.createSCOTCHArch()
            if ok == False:
                return False
            ok = self.scotchLib.populatePartitionArchitecture(self.numPartitions)
            return ok
        return False

    def initStrategy(self):
        if self.scotchLib.isLoaded():
            #self.numPartitions = numPartitions
            ok = self.scotchLib.createStrategy()
            if ok == False:
                return False
            ok = self.scotchLib.setStrategyGraphMapBuild(self.strategyFlag, self.numPartitions, self.kbalval)
            if ok == False:
                return False
            return ok
            ok = self.scotchLib.setStrategyFlags(self.strategyOptions)
            if ok == False:
                return False
            return True
        return False

    def loadGraph(self, scotchData, skipGraphValidStep = False):
        if isValidScotchData(scotchData) == False:
            print("loadGraph: not Valid Scotch Data")
            return False
        else:
            self.scotchData = scotchData

        if self.scotchLib.isLoaded():
            ok = self.scotchLib.createSCOTCHGraph()
            if ok == False:
                print("loadGraph: cannot createSCOTCHGraph")
                return False
            ok  = self.scotchLib.buildSCOTCHGraphFromData(self.scotchData)
            if ok == False:
                print("loadGraph: cannot buildSCOTCHGraphFromData")
                return False

            if skipGraphValidStep == True:
                return ok

            ok = self.scotchLib.scotchGraphValid()
            if ok == False:
                print("loadGraph: scotchGraphValid returned false")
            return ok
        return False

    def graphMap(self):
        if self.scotchLib.isLoaded():

            #arr = (ctypes.c_int * 1000)(*self.scotchData.parttab)
            #return self.scotchLib.graphMap(arr)

            return self.scotchLib.graphMap(self.scotchData._parttab)
        return False

    def graphMapFixed(self):
        if self.scotchLib.isLoaded():
            return self.scotchLib.graphMapFixed(self.scotchData._parttab)
        return False
