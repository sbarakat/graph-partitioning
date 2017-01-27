# Setup Instructions for SCOTCH


## 1) Download SCOTCH python library

clone: https://github.com/cmmin/csap-graphpartitioning

The default folder structure should be:

```
path/to/graph-partitioning/...
path/to/csap-graphpartitioning/...
```

## 2) Linux: setup SCOTCH 6.0.4

On linux, you must ensure that the SCOTCH 6.0.4 shared library is installed on your machine at a default path. You should use the version that we have compiled as it has a relative-path search for libscotcherr.so.

The simplest way to set this up is to:

```
$ cd path/to/csap-graphpartitioning/tools/scotch

# Installs SCOTCH 6.0.4 on linux at /usr/local/lib/scotch_604
$ ./install_scotch_linux.sh

```

## 3) Setting up ```config.py```

Please copy and paste config_template.py to config.py. The file config.py is ignored by git, so any local settings won't be version controlled. Please update config_template.py if any major changes (ie. addition/removal of variables) is made.

## 4) Testing the setup

To test the setup, try loading config.py:

```
$ python3
>>> import config

// If all is setup correctly, it should print out:
SCOTCH Environment valid.
SCOTCH python bindings were loaded correctly from ../csap-graphpartitioning/src/python
SCOTCH Library was located successfully at ../csap-graphpartitioning/tools/scotch/lib/macOS/libscotch.dylib
```
