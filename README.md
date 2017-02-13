# Graph Partitioning

This repository contains research into the use of graph partitioning algorithms for assigning people shelters based on their social networks in disaster areas. Many of the ideas that appear here were done in collaboration with Nathaniel Douglass.

## Initial Setup

The following setup instructions will get the prerequisites installed for running the Jupyter Notebook. These steps assume Ubuntu 16.04 (Xenial Xerus).

    # ensure these are removed
    sudo apt-get purge ipython3 ipython3-notebook

    # requirements for python3 notebook
    sudo apt-get install python3-dev python3-pip build-essential libzmq3-dev libpng-dev libjpeg8-dev libfreetype6-dev

    # clone git repo and build
    git clone https://github.com/sbarakat/graph-partitioning.git
    cd graph-partitioning/

    # not needed on Ubuntu Xenial
    sudo add-apt-repository ppa:marutter/rdev
    sudo apt-get update

    # install R 3.2
    sudo apt-get install r-base r-base-dev

    # virtualenv setup
    sudo pip3 install virtualenv
    virtualenv -p python3 env
    source env/bin/activate
    pip3 install -r requirements.txt

### Graph Metrics

The last section of the Jupyter Notebook produces metrics to check the quality of the partitioning algorithm. The following setup is only needed if these metrics are desired.

#### Requirements for MaxPerm

    # for Ubuntu 14.04
    sudo apt-get install libigraph0 libigraph0-dev
    # for Ubuntu 16.04
    sudo apt-get install libigraph0v5 libigraph0-dev
    cd bin/MaxPerm/
    gcc Main.c MaxPerm.c -I/usr/include/igraph/ -ligraph -lm -o MaxPerm

#### Requirements for OSLOM2

    cd bin/
    wget http://www.oslom.org/code/OSLOM2.tar.gz
    tar -xvzf OSLOM2.tar.gz
    rm OSLOM2.tar.gz
    cd OSLOM2/
    ./compile_all.sh

#### Requirements for ComQualityMetric

    cd bin/ComQualityMetric/
    javac OverlappingCommunityQuality.java
    javac CommunityQuality.java

## Run the notebook

    cd graph-partitioning/
    source env/bin/activate
    ipython3 notebook graph-partitioning-fennel.ipynb

## Acknowledgements

Further information and the source code used in this repository can be found below:

* Original notebook adapted from Justin Vincent's research on [Graph Partitioning](http://algorithmshop.com/20131213-graph-partitioning.html). Released as public domain and available on [GitHub](https://github.com/justinvf/algorithmshop/blob/master/20131213-graph-partitioning/20131213-graph-partitioning.ipynb).
* [Fennel: Streaming Graph Partitioning for Massive Scale Graphs](https://www.microsoft.com/en-us/research/publication/fennel-streaming-graph-partitioning-for-massive-scale-graphs/). A Microsoft Technical Report by Charalampos E. Tsourakakis, Christos Gkantsidis, Bozidar Radunovic, Milan Vojnovic.
* [OSLOM](http://www.oslom.org/)
* [ComQualityMetric](https://github.com/chenmingming/ComQualityMetric) by Mingming Chen.

## Author

Sami Barakat (<sami@sbarakat.co.uk>)

Licensed under the MIT license.  See the [LICENSE](https://github.com/sbarakat/graph-partitioning/blob/master/LICENSE) file for further details.
