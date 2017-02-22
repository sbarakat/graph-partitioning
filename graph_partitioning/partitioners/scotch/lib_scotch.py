import ctypes # used for accessing the dynamic library

import graph_partitioning.partitioners.utils as putils # used for some of the utilities functions

class LibScotch(putils.CLibInterface):
    def __init__(self, libraryPath = None):
        super().__init__(libraryPath=libraryPath)

    def _getDefaultLibPath(self):
        return putils.defaultSCOTCHLibraryPath()

    def _loadLibraryFunctions(self):
        # *****************
        # structures & data
        # *****************

        # These describe the type of object to be created
        self.SCOTCH_Arch = ctypes.c_double*128
        self.SCOTCH_Graph = ctypes.c_double*128
        self.SCOTCH_Strat = ctypes.c_double*128

        # These store the scotch data objects (ie. graph = SCOTCH_Graph())
        self.architecture = None
        self.graph = None
        self.strategy = None

        self.SCOTCH_version = self.clib.SCOTCH_version
        self.SCOTCH_version.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]

        # SCOTCH_archAlloc
        self.SCOTCH_archAlloc = self.clib.SCOTCH_archAlloc
        #self.SCOTCH_archAlloc.argtypes = [ None ]

        # SCOTCH_archInit
        self.SCOTCH_archInit = self.clib.SCOTCH_archInit
        self.SCOTCH_archInit.argtypes = [ctypes.POINTER(self.SCOTCH_Arch)]

        # SCOTCH_archExit
        self.SCOTCH_archExit = self.clib.SCOTCH_archExit
        self.SCOTCH_archExit.argtypes = [ctypes.POINTER(self.SCOTCH_Arch)]

        # SCOTCH_archCmplt - builds architecture for partitioning
        self.SCOTCH_archCmplt = self.clib.SCOTCH_archCmplt
        self.SCOTCH_archCmplt.argtypes = [ctypes.POINTER(self.SCOTCH_Arch), ctypes.c_int]

        # SCOTCH_graphAlloc
        self.SCOTCH_graphAlloc = self.clib.SCOTCH_graphAlloc
        #self.SCOTCH_graphAlloc.argtypes = [ None ]

        # SCOTCH_graphInit
        self.SCOTCH_graphInit = self.clib.SCOTCH_graphInit
        self.SCOTCH_graphInit.argtypes = [ctypes.POINTER(self.SCOTCH_Graph)]

        # SCOTCH_graphExit
        self.SCOTCH_graphExit = self.clib.SCOTCH_graphExit
        self.SCOTCH_graphExit.argtypes = [ctypes.POINTER(self.SCOTCH_Graph)]

        # SCOTCH_graphCheck
        self.SCOTCH_graphCheck = self.clib.SCOTCH_graphCheck
        self.SCOTCH_graphCheck.argtypes = [ctypes.POINTER(self.SCOTCH_Graph)]

        # SCOTCH_graphBuild
        self.SCOTCH_graphBuild = self.clib.SCOTCH_graphBuild
        self.SCOTCH_graphBuild.argtypes = [
            ctypes.POINTER(self.SCOTCH_Graph), ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p
        ]

        # SCOTCH_stratAlloc
        self.SCOTCH_stratAlloc = self.clib.SCOTCH_stratAlloc
        #self.SCOTCH_stratAlloc.argtypes = [ None ]

        # SCOTCH_stratInit
        self.SCOTCH_stratInit = self.clib.SCOTCH_stratInit
        self.SCOTCH_stratInit.argtypes = [ctypes.POINTER(self.SCOTCH_Strat)]

        self.SCOTCH_stratExit = self.clib.SCOTCH_stratExit
        self.SCOTCH_stratExit.argtypes = [ctypes.POINTER(self.SCOTCH_Strat)]

        self.SCOTCH_stratGraphMap = self.clib.SCOTCH_stratGraphMap
        self.SCOTCH_stratGraphMap.argtypes = [ctypes.POINTER(self.SCOTCH_Strat), ctypes.c_char_p]

        self.SCOTCH_stratGraphMapBuild = self.clib.SCOTCH_stratGraphMapBuild
        self.SCOTCH_stratGraphMapBuild.argtypes = [ctypes.POINTER(self.SCOTCH_Strat), ctypes.c_int, ctypes.c_int, ctypes.c_double]

        # MAPPING Functions
        self.SCOTCH_graphMap = self.clib.SCOTCH_graphMap
        self.SCOTCH_graphMap.argtypes = [ctypes.POINTER(self.SCOTCH_Graph), ctypes.POINTER(self.SCOTCH_Arch), ctypes.POINTER(self.SCOTCH_Strat), ctypes.c_void_p]

        self.SCOTCH_graphMapFixed = self.clib.SCOTCH_graphMapFixed
        self.SCOTCH_graphMapFixed.argtypes = [ctypes.POINTER(self.SCOTCH_Graph), ctypes.POINTER(self.SCOTCH_Arch), ctypes.POINTER(self.SCOTCH_Strat), ctypes.c_void_p]

    def isLoaded(self):
        if self.clib is None:
            return False
        return True

    def version(self):
        major_ptr = ctypes.c_int(0)
        relative_ptr = ctypes.c_int(0)
        patch_ptr = ctypes.c_int(0)

        ret = self.SCOTCH_version(major_ptr, relative_ptr, patch_ptr)
        return "{}.{}.{}".format(major_ptr.value, relative_ptr.value, patch_ptr.value)

    def createSCOTCHArch(self):
        #self.SCOTCH_Arch = self.SCOTCH_archAlloc()
        #print(self.SCOTCH_Arch)
        self.architecture = self.SCOTCH_Arch()
        ret = self.SCOTCH_archInit(self.architecture)
        if(ret == 0):
            return True
        return False

    def deleteSCOTCHStrat(self):
        self.SCOTCH_stratExit(self.strategy)
        del self.strategy
        self.strategy = None

    def deleteSCOTCHArch(self):
        self.SCOTCH_archExit(self.architecture)
        del self.architecture
        self.architecture = None

    def populatePartitionArchitecture(self, numPartitions):
        if(self.architecture is None):
            return False

        if(isinstance(numPartitions, int)):
            ret = self.SCOTCH_archCmplt(self.architecture, numPartitions)
            if(ret == 0):
                return True
        return False

    def createSCOTCHGraph(self):
        #self.SCOTCH_Graph = self.SCOTCH_graphAlloc()
        self.graph = self.SCOTCH_Graph()
        ret = self.SCOTCH_graphInit(self.graph)
        if(ret == 0):
            return True
        return False

    def buildSCOTCHGraphFromData(self, scotchData):
        #if isinstance(scotchData, scotchio.ScotchGraphArrays) == False:
        #    return False

        if self.graph is None:
            if(self.createSCOTCHGraph() == False):
                return False

        if scotchData._vlbltab is None:
            success = self.SCOTCH_graphBuild(self.graph, scotchData.baseval, scotchData.vertnbr, scotchData._verttab.ctypes, 0, scotchData._velotab.ctypes, 0, scotchData.edgenbr, scotchData._edgetab.ctypes, scotchData._edlotab.ctypes)
        else:
            #print('SCOTCH.py, using vlbltab array')
            success = self.SCOTCH_graphBuild(self.graph, scotchData.baseval, scotchData.vertnbr, scotchData._verttab.ctypes, 0, scotchData._velotab.ctypes, scotchData._vlbltab.ctypes, scotchData.edgenbr, scotchData._edgetab.ctypes, scotchData._edlotab.ctypes)

        if success == 0:
            return True
        return False


    def deleteSCOTCHGraph(self):
        # TODO write test for this
        self.SCOTCH_graphExit(self.graph)
        del self.graph
        self.graph = None

    def scotchGraphValid(self):
        # TODO write test for this
        ret = self.SCOTCH_graphCheck(self.graph)
        if(ret == 0):
            return True
        return False

    def createStrategy(self):
        self.strategy = self.SCOTCH_Strat()
        ret = self.SCOTCH_stratInit(self.strategy)
        if ret == 0:
            return True
        return False

    def setStrategyGraphMapBuild(self, straval, partitionNbr, kbalval = 0.1):
        ret = self.SCOTCH_stratGraphMapBuild(self.strategy, straval, partitionNbr, kbalval)
        if ret == 0:
            return True
        return False

    def setStrategyFlags(self, strategyFlags):
        if(isinstance(strategyFlags, str) == False):
            strategyFlags = ''
        # Note: must encode the string as that returns a bytecode equivalent
        success = self.SCOTCH_stratGraphMap(self.strategy, strategyFlags.encode('utf-8'))
        if(success == 0):
            return True
        return False

    def createSCOTCHGraphMapStrategy(self, strategyFlags):
        #self.strategy = self.SCOTCH_stratAlloc()
        self.strategy = self.SCOTCH_Strat()
        ret = self.SCOTCH_stratInit(self.strategy)
        if(ret == 0):
            if(isinstance(strategyFlags, str) == False):
                strategyFlags = ''
            # Note: must encode the string as that returns a bytecode equivalent
            success = self.SCOTCH_stratGraphMap(self.strategy, strategyFlags.encode('utf-8'))
            if(success == 0):
                return True
        return False

    def graphMap(self, parttab):

        ret = self.SCOTCH_graphMap(self.graph, self.architecture, self.strategy, parttab.ctypes)
        if ret == 0:
            return True
        return False

    def graphMapFixed(self, parttab):
        ret = self.SCOTCH_graphMapFixed(self.graph, self.architecture, self.strategy, parttab.ctypes)
        if ret == 0:
            return True
        return False
