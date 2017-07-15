# Graph Partitioning Setup
```Python```

## Third-party tools used
```Scotch```

```PaToH```
```Jupyter Notebook```


## Python packages that are required

scipy, networkx, numpy, louvain community, setuptools


### Infomap

http://www.mapequation.org/code.html#Examples-with-python

```Infomap (C++/python)```: requires SWIG, Setuptools for python version. C++ binary requires clang or gcc. To build infomap with python3 instead of python 2.7, you must change the Makefile in the examples/python folder so that the following lines read:

```PY_BUILD_DIR = $(INFOMAP_DIR)/build/py3```

```cd $(INFOMAP_DIR) && $(MAKE) python3```

rather than:

```PY_BUILD_DIR = $(INFOMAP_DIR)/build/py```

```cd $(INFOMAP_DIR) && $(MAKE) python```
