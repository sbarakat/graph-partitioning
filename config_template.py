import os
import sys
from enum import Enum

from utilities.system_utils import getOS, OS

# Enumeration that stores different types of algorithms
class Partitioners(Enum):
    FENNEL = 1
    SCOTCH = 2
    PATOH = 3

PREDICTION_MODEL_ALGORITHM = Partitioners.SCOTCH

# SCOTCH
ENABLE_SCOTCH = True
SCOTCH_ENVIRONMENT_VALID = True # assume valid
SCOTCH_PYLIB_REL_PATH = '../csap-libraries-porting/src/python'

SCOTCH_HAS_GRAPHMAPFIXED = True

SCOTCH_LIB_PATH = ''
if getOS() == OS.macOS:
    SCOTCH_LIB_PATH = '../csap-libraries-porting/tools/scotch/lib/macOS/libscotch.dylib'
elif getOS() == OS.linux:
    SCOTCH_LIB_PATH = '../csap-libraries-porting/tools/scotch/lib/linux/libscotch.so'

if ENABLE_SCOTCH == True:
    print("SCOTCH Enabled.")
    # allow python to find modules at the relative path
    sys.path.insert(0, SCOTCH_PYLIB_REL_PATH)

    # try importing scotch
    try:
        import scotch
    except ImportError as err:
        print(err)
        print("Could not load SCOTCH, check that SCOTCH_PYLIB_REL_PATH is set correctly in config.py")
        SCOTCH_ENVIRONMENT_VALID = False

    # check if the library is present
    if os.path.isfile(SCOTCH_LIB_PATH) == False:
        # check if this in linux and try and find it on the default path
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

if ENABLE_SCOTCH == False or SCOTCH_ENVIRONMENT_VALID == False:
    print("Enabling FENNEL algorithm, SCOTCH environment invalid OR Scotch not enabled.")
    PREDICTION_MODEL_ALGORITHM = Partitioners.FENNEL
