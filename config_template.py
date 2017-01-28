import os
import sys
from enum import Enum

# Enumeration that stores different types of algorithms
class Partitioners(Enum):
    FENNEL = 1
    SCOTCH = 2
    PATOH = 3

# set which algorithm is run for the PREDICTION MODEL
PREDICTION_MODEL_ALGORITHM = Partitioners.FENNEL
ASSIGNMENT_MODEL_ALGORITHM = Partitioners.FENNEL

# if set to True, we run all the prediction models and store them separately
RUN_ALL_PREDICTION_MODEL_ALGORITHMS = True

# SCOTCH SETTINGS
ENABLE_SCOTCH = True # if set to True, SCOTCH parameters get loaded

SCOTCH_ENVIRONMENT_VALID = True # assume that the system is configured correctly for SCOTCH
SCOTCH_PYLIB_REL_PATH = '../csap-graphpartitioning/src/python' # relative path to the SCOTCH python modules

# depending on scotch version that is available, SCOTCH_graphMapFixed may be available or not
SCOTCH_HAS_GRAPHMAPFIXED = True

# path to the scotch shared library
SCOTCH_LIB_PATH = ''

if ENABLE_SCOTCH == True:
    # allow python to find modules at the relative path
    sys.path.insert(0, SCOTCH_PYLIB_REL_PATH)

    # import some utilities from the SCOTCH module to test the setup
    from utilities.system_utils import getOS, OS

    # try importing scotch
    try:
        import scotch

        # set SCOTCH_LIB_PATH to the default path
        SCOTCH_LIB_PATH = scotch.scotch.defaultLibraryPath()

    except ImportError as err:
        print(err)
        print("Could not load SCOTCH, check that SCOTCH_PYLIB_REL_PATH is set correctly in config.py")
        SCOTCH_ENVIRONMENT_VALID = False

    # OPTIONAL: override the default SCOTCH_LIB_PATH
    if getOS() == OS.macOS:
        pass
        #SCOTCH_LIB_PATH = '../csap-graphpartitioning/tools/scotch/lib/macOS/libscotch.dylib'
    elif getOS() == OS.linux:
        pass
        #SCOTCH_LIB_PATH = '/usr/local/lib/scotch_604/libscotch.so'

    # check if the library file is present
    if os.path.isfile(SCOTCH_LIB_PATH) == False:
        # check if the OS is linux and try and find the library on the default lib path
        if getOS() == OS.linux:
            SCOTCH_LIB_PATH = '/usr/lib/libscotch-6.0.4.so'
            if os.path.isfile(SCOTCH_LIB_PATH) == False:
                # try version 5.1
                SCOTCH_LIB_PATH = '/usr/lib/libscotch-5.1.so'
                if(os.path.isfile(SCOTCH_LIB_PATH)) == False:
                    print("** Could not locate the SCOTCH library file at:", SCOTCH_LIB_PATH)
                    SCOTCH_ENVIRONMENT_VALID = False
                else:
                    # disable graphMapFixed as not present in 5.1
                    SCOTCH_HAS_GRAPHMAPFIXED = False
        else:
            print("** Could not locate the SCOTCH library file at:", SCOTCH_LIB_PATH)
            SCOTCH_ENVIRONMENT_VALID = False

    if SCOTCH_ENVIRONMENT_VALID:
        print("SCOTCH Environment valid.\nSCOTCH python bindings were loaded correctly from", SCOTCH_PYLIB_REL_PATH,"\nSCOTCH Library was located successfully at", SCOTCH_LIB_PATH)
    else:
        print("SCOTCH Environment NOT VALID.\nTried loading SCOTCH python bindings from", SCOTCH_PYLIB_REL_PATH,"\nTried locating SCOTCH Library at", SCOTCH_LIB_PATH, '\n Try checking these paths in config.py')

# If SCOTCH wasn't loaded properly, default algorithm for PREDICTION_MODEL is Fennel
if ENABLE_SCOTCH == False or SCOTCH_ENVIRONMENT_VALID == False:
    print("Enabling FENNEL algorithm, SCOTCH environment invalid OR Scotch not enabled.")
    PREDICTION_MODEL_ALGORITHM = Partitioners.FENNEL
    ASSIGNMENT_MODEL_ALGORITHM = Partitioners.FENNEL
