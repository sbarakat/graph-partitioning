import ctypes

import graph_partitioning.partitioners.utils as putils
from graph_partitioning.partitioners.patoh.parameters import PATOHParameters

class LibPatoh(putils.CLibInterface):
    def __init__(self, libraryPath = None):
        super().__init__(libraryPath=libraryPath)

    def _getDefaultLibPath(self):
        return putils.defaultPATOHLibraryPath()

    def _loadLibraryFunctions(self):
        self.PATOH_Version = self.clib.Patoh_VersionStr
        self.PATOH_Version.restype = (ctypes.c_char_p)

        self.PATOH_InitializeParameters = self.clib.Patoh_Initialize_Parameters
        self.PATOH_InitializeParameters.argtypes = (ctypes.POINTER(PATOHParameters), ctypes.c_int, ctypes.c_int)

        self.PATOH_checkUserParameters = self.clib.Patoh_Check_User_Parameters
        self.PATOH_checkUserParameters.argtypes = (ctypes.POINTER(PATOHParameters), ctypes.c_int)

        self.PATOH_Alloc = self.clib.Patoh_Alloc
        self.PATOH_Alloc.argtypes = (ctypes.POINTER(PATOHParameters), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)

        self.PATOH_Part = self.clib.Patoh_Part

        self.PATOH_Part.argtypes = (ctypes.POINTER(PATOHParameters), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)


        self.PATOH_Free = self.clib.Patoh_Free

        self.cfree = self.clib.free
        self.cfree.argtypes = (ctypes.c_void_p,)

    def version(self):
        return self.PATOH_Version().decode('utf-8')

    def initializeParameters(self, patohData, num_partitions = 2):
        if(isinstance(num_partitions, int) == False):
            num_partitions = 2

        patohData.params = PATOHParameters()
        ok = self.PATOH_InitializeParameters(ctypes.byref(patohData.params), 1, 0)
        if(ok == 0):
            patohData.params._k = num_partitions
            return True
        else:
            patohData.params = None
            return False

    def checkUserParameters(self, patohData, verbose = True):
        if (isinstance(patohData.params, PATOHParameters) == False):
            print('Cannot check parameters as params is not of type PATOHParameters')
            return False

        # check verbosity mode
        v = 0
        if verbose == True:
            v = 1

        # perform parameter check
        ok = self.PATOH_checkUserParameters(ctypes.byref(patohData.params), v)
        if(ok == 0):
            print('User Parameters Valid')
            return True
        else:
            print('Error in the user parameters. Use verbose mode for greater details.')
            return False

    def alloc(self, patohData):
        #if (isinstance(patohData, patdata.PatohData) == False):
        #        return False

        #PPaToH_Parameters pargs, int _c, int _n, int _nconst, int *cwghts, int *nwghts, int *xpins, int *pins
        ok = self.PATOH_Alloc(ctypes.byref(patohData.params), patohData._c, patohData._n, patohData._nconst, patohData._cwghts.ctypes, patohData._nwghts.ctypes, patohData._xpins.ctypes, patohData._pins.ctypes)
        if (ok == 0):
            return True
        return False

    def part(self, patohData):

        '''
        int PaToH_Part(PPaToH_Parameters pargs, int _c, int _n, int _nconst, int useFixCells,
               int *cwghts, int *nwghts, int *xpins, int *pins, float *targetweights,
               int *partvec, int *partweights, int *cut);


        '''
        cut_val = ctypes.c_int(patohData.cut)
        cut_addr = ctypes.addressof(cut_val)

        ok = self.PATOH_Part(ctypes.byref(patohData.params), patohData._c, patohData._n, patohData._nconst, patohData.useFixCells, patohData._cwghts.ctypes, patohData._nwghts.ctypes, patohData._xpins.ctypes, patohData._pins.ctypes, patohData._targetweights.ctypes, patohData._partvec.ctypes, patohData._partweights.ctypes, cut_addr)

        if (ok == 0):
            # get value back
            patohData.cut = cut_val

            return True
        return False

    def free(self, patohData):
        #self.cfree(patohData._cwghts.ctypes)
        #self.cfree(patohData._nwghts.ctypes)
        #self.cfree(patohData._xpins.ctypes)
        #self.cfree(patohData._pins.ctypes)
        #self.cfree(patohData._partweights.ctypes)
        #self.cfree(patohData._partvec.ctypes)

        ok = self.PATOH_Free()
        if ok == 0:
            return True
        return False
