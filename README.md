
Initial setup notes

    # ensure these are removed
    sudo apt-get purge ipython3 ipython3-notebook

    # requirements for python3 notebook
    sudo apt-get install python3-dev python3-pip build-essential libzmq3-dev libpng-dev libjpeg8-dev libfreetype6-dev

    # clone git repo and build
    git clone https://github.com/sbarakat/algorithmshop-graph-partitioning.git
    cd algorithmshop-graph-partitioning/

    # virtualenv setup
    sudo pip3 install virtualenv
    virtualenv -p python3 env
    source env/bin/activate
    pip3 install -r requirements.txt

Run the notebook:

    cd algorithmshop-graph-partitioning/
    source env/bin/activate
    ipython3 notebook graph-partitioning-fennel.ipynb

Requirements for MaxPerm

    sudo apt-get install libigraph0 libigraph0-dev
    cd bin/MaxPerm/
    gcc Main.c MaxPerm.c -I/usr/include/igraph/ -ligraph -lm -o MaxPerm

Requirements for OSLOM2

    cd bin/
    wget http://www.oslom.org/code/OSLOM2.tar.gz
    tar -xvzf OSLOM2.tar.gz
    rm OSLOM2.tar.gz
    cd OSLOM2/
    ./compile_all.sh

Requirements for ComQualityMetric

    cd bin/ComQualityMetric/
    javac OverlappingCommunityQuality.java
    javac CommunityQuality.java

