
Ensure these are removed:

    sudo apt-get purge ipython3 ipython3-notebook

Install requirements for iPython3 notebook

    sudo apt-get install python3-dev python3-pip build-essential libzmq3-dev libpng-dev libjpeg8-dev libfreetype6-dev
    sudo pip3 install virtualenv
    cd /home/nkd26/Desktop/algorithmshop-master/
    virtualenv -p python3 env
    source env/bin/activate
    pip3 install ipython[all]
    pip3 install numpy
    pip3 install cython
    pip3 install matplotlib

Run the notebook:

    cd /home/nkd26/Desktop/algorithmshop-master/
    source env/bin/activate
    ipython3 notebook graph-partitioning.ipynb

Requirements for networkit

    sudo apt-get install libblas-dev liblapack-dev libatlas-base-dev gfortran
    pip install scipy
    pip install tabulate
    pip install pandas
    pip install seaborn
    pip install networkit

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

