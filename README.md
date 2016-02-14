
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

