import os
import platform
import ctypes
from pathlib import Path

import numpy as np

def minPartitionCounts(assignments, num_partitions):
    ''' Makes a count of each node assigned to each partition. Returns the partition with the smallest count
    and a dictionary indexing the counts for each partition: partitions[partition_id] = totalNodesInPartition '''
    partitions = {}
    for i in range(num_partitions):
        partitions[i] = 0

    for partition in assignments:
        if partition < 0:
            continue
        if partition in partitions:
            partitions[partition] += 1
        else:
            partitions[partition] = 1

    minCount = len(assignments) + 1
    minCountPartition = -1
    for key in list(partitions.keys()):
        if partitions[key] < minCount:
            minCount = partitions[key]
            minCountPartition = key

    if minCountPartition < 0:
        minCountPartition = pickRandPartition(num_partitions)

    return (minCountPartition, partitions)

def pickRandPartition(num_partitions):
    ''' Picks a random number in range 0 to num_partitions - 1 '''
    return randNumInRange(num_partitions - 1)

def randNumInRange(maxRangeVal):
    ''' Picks a random number between 0 and maxRangeVal '''
    if maxRangeVal <= 0:
        return 0

    val = 0
    while True:
        val = int(np.floor(abs(np.random.randn(1)[0]) * maxRangeVal))
        if val >= 0 and val <= maxRangeVal:
            break
    return val

def genArray(arr_length, defaultVal = 0):
    ''' Generates an array list of length arr_length filled of values = defaultVal '''
    arr = []
    for i in range(0, arr_length):
        arr.append(defaultVal)
    if arr_length != len(arr):
        print('genArr error in generating number of array')
    return arr

def exportArrayToNumpyArray(array, dtype=np.int32):
    ''' Converts an array list to a numpy array '''
    if (array is None) or (isinstance(array, list) == False):
        array = []
    return np.asanyarray(array, dtype=dtype)

def getOS():
    ''' Returns the current operating system as macOS or linux '''
    if 'Darwin' in platform.system():
        return 'macOS'
    elif 'Linux' in platform.system():
        return 'linux'
    else:
        return ''

def isLinux():
    ''' Returns True if this system is Linux '''
    if getOS() == 'linux':
        return True
    return False

def isMacOS():
    ''' Returns True if this system is macOS '''
    if getOS() == 'macOS':
        return True
    return False

def defaultSCOTCHLibraryPath():
    ''' Returns the default path to the SCOTCH dynamic library for macOS and linux '''
    if isLinux():
        return '/usr/local/lib/scotch_604/libscotch.so'
    if isMacOS():
        return os.path.join(str(Path(__file__).parents[2]), 'libs/scotch/macOS/libscotch.dylib')
    return ''

def defaultPATOHLibraryPath():
    ''' Returns the default path to the PaToH dynamic library for macOS and linux '''
    if isLinux():
        return os.path.join(str(Path(__file__).parents[2]), 'libs/patoh/lib/linux/libpatoh.so')
    if isMacOS():
        return os.path.join(str(Path(__file__).parents[2]), 'libs/patoh/lib/macOS/libpatoh.dylib')
    return ''

class CLibLoadException(Exception):
    ''' An exception caused by a problem loading the library '''


class CLibInterface:
    ''' Base class that acts as an interface to load C/C++ Dynamic Libraries using ctypes '''

    def __init__(self, libraryPath = None):
        if libraryPath is None:
            libraryPath = self._getDefaultLibPath()

        self.libraryPath = libraryPath
        self.clib = None

    def _getDefaultLibPath(self):
        ''' Base function to get the default path for the library. Must be re-implemented by child classes '''
        return ''

    def _loadLibraryFunctions(self):
        ''' Base function to load the library's desired functions. Must be re-implemented by child classes '''
        return True

    def libIsLoaded(self):
        if self.clib is None:
            return False
        return True

    def load(self):
        ''' Base function that loads a ctypes Dynamic Library, then loads the library functions. Exception safe. '''
        try:
            self.clib = ctypes.cdll.LoadLibrary(self.libraryPath)
            self._loadLibraryFunctions()
            return True
        except Exception as err:
            print('Error loading library at path', self.libraryPath, '; error:', err)
            self.clib = None
