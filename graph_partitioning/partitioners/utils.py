import os
import platform
import ctypes
from pathlib import Path

import numpy as np


def genArray(arr_length, defaultVal = 0):
    arr = []
    for i in range(0, arr_length):
        arr.append(defaultVal)
    if arr_length != len(arr):
        print('genArr error in generating number of array')
    return arr

def exportArrayToNumpyArray(array, dtype=np.int32):
    if array is None:
        array = []
    return np.asanyarray(array, dtype=dtype)

def getOS():
    if 'Darwin' in platform.system():
        return 'macOS'
    elif 'Linux' in platform.system():
        return 'linux'
    else:
        return ''

def isLinux():
    if getOS() == 'linux':
        return True
    return False

def isMacOS():
    if getOS() == 'macOS':
        return True
    return False

def defaultSCOTCHLibraryPath():
    if isLinux():
        return '/usr/local/lib/scotch_604/libscotch.so'
    if isMacOS():
        return os.path.join(str(Path(__file__).parents[2]), 'libs/scotch/macOS/libscotch.dylib')
    return ''

def defaultPATOHLibraryPath():
    if isLinux():
        return os.path.join(str(Path(__file__).parents[2]), 'libs/patoh/lib/linux/libpatoh.so')
    if isMacOS():
        return os.path.join(str(Path(__file__).parents[2]), 'libs/patoh/lib/macOS/libpatoh.dylib')
    return ''

class CLibLoadException(Exception):
    ''' An exception caused by a problem loading the library '''


class CLibInterface:
    def __init__(self, libraryPath = None):
        if libraryPath is None:
            libraryPath = self._getDefaultLibPath()

        self.libraryPath = libraryPath
        self.clib = None

    def _getDefaultLibPath(self):
        return ''

    def _loadLibraryFunctions(self):
        return True

    def load(self):
        try:
            self.clib = ctypes.cdll.LoadLibrary(self.libraryPath)
            self._loadLibraryFunctions()
            return True
        except Exception as err:
            print('Error loading library at path', self.libraryPath, '; error:', err)
