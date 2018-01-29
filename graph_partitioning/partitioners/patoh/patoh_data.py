import numpy as np
import networkx as nx

import graph_partitioning.partitioners.utils as putils

class PatohData:
    ''' This class stores all the data and arrays used by PaToH during partitioning '''

    def __init__(self):
        self.initialize()

    def initialize(self):
        self._c = None
        self._n = None
        self._nconst = 1
        self.cwghts = []
        self.nwghts = []
        self.xpins = []
        self.pins = []
        self.partvec = []
        self.useFixCells = 0 # 0 assumes no partitions assigned
        self.cut = 0
        self.targetweights = []
        self.partweights = []

        self.params = None

        # exported arrays
        self._cwghts = None
        self._nwghts = None
        self._xpins = None
        self._pins = None
        self._partvec = None

        self._targetweights = None
        self._partweights = None

    def debugPrint(self):
        print('_c', self._c)
        print('_n', self._n)
        print('_nconst', self._nconst)
        print('cwghts', self.cwghts)
        print('nwghts', self.nwghts)
        print('xpins', self.xpins)
        print('pins', self.pins)
        print('partvec', self.partvec)
        print('useFixCell', self.useFixCells)
        print('cut', self.cut)
        print('targetweights', self.targetweights)
        print('partweights', self.partweights)

    def debugPrintExport(self):
        print('_cwghts', self._cwghts, self._cwghts.ctypes)
        print('_nwghts', self._nwghts, self._nwghts.ctypes)
        print('_xpins', self._xpins, self._xpins.ctypes)
        print('_pins', self._pins, self._pins.ctypes)
        print('_partvec', self._partvec, self._partvec.ctypes)

        print('_targetweights', self._targetweights, self._targetweights.ctypes)
        print('_partweights', self._partweights, self._partweights.ctypes)

    def fromNetworkxGraph(self, G, num_partitions, partvec = None, hyperedgeExpansionMode='no_expansion'):
        ''' Populates the PaToH data from a networkx graph '''

        if(isinstance(G, nx.Graph) == False):
            return False

        cliques = self._getGraphCliques(G)
        if cliques is None:
            return False

        self._c = G.number_of_nodes()
        self._n = len(cliques)

        #xpins stores the index starts of each net (clique)
        #pins stores the node ides in each clique indexed by xpins
        self.xpins = putils.genArray(self._n + 1, 0)
        self.pins = []

        # _nconst = number of weights for each vertex/cell
        # ASSUME _nconst = 1
        # node weights = cwhgts
        self.cwghts = putils.genArray(self._c * self._nconst, 1)

        # edge weights need to be converted to nwghts
        self.nwghts = putils.genArray(self._n, 1)

        for cliqueID, clique in enumerate(cliques):
            self.xpins[cliqueID] = len(self.pins)

            node_weights = []
            clique_edge_weights = []
            clique_node_weights = []

            edges_added = []

            for node in clique:
                # set the node weight
                try:
                    weight = G.node[node]['weight']
                    #node_weights.append(weight)
                    clique_node_weights.append(weight)
                    self.cwghts[node] = weight
                except Exception as err:
                    #print('node weight exception', err)
                    #node_weights.append(1)
                    pass

                for edgeID in G.neighbors(node):
                    # make sure each edge is counted only once
                    edge_str = str(node)
                    if edgeID > node:
                        edge_str = edge_str + '_' + str(edgeID)
                    else:
                        edge_str = str(edgeID) + '_' + edge_str

                    if edge_str in edges_added:
                        continue
                    else:
                        edges_added.append(edge_str)

                    try:
                        eWeight = G.edge[node][edgeID]['weight']
                        clique_edge_weights.append(eWeight)
                    except Exception as err:
                        print('clique edge error', err)

                self.pins.append(node)

            # compute clique edge expansion
            if '_complete' in hyperedgeExpansionMode:
                hyperedgeWeight = self._hyperedgeExpansionComplete(clique_node_weights, hyperedgeExpansionMode)
            else:
                hyperedgeWeight = self._hyperedgeExpansion(clique_node_weights, hyperedgeExpansionMode)

            #print('hyperedge', len(clique), hyperedgeExpansionMode, hyperedgeWeight)

            self.nwghts[cliqueID] = hyperedgeWeight

        # add last ID
        self.xpins[self._n] = len(self.pins)

        if partvec is not None:
            self.useFixCells = 1
            self.partvec = partvec

        self._setTargetPartitionWeights(num_partitions)

        self._exportArrays()

    def _getGraphCliques(self, G):
        if(isinstance(G, nx.Graph) == False):
            return None
        return list(nx.find_cliques(G))

    def _setTargetPartitionWeights(self, num_partitions):
        target = 1.0 / float(num_partitions)
        for k in range(0, num_partitions):
            self.targetweights.append(target)

        self.partweights = putils.genArray(num_partitions * self._nconst, 0)

    def _exportArrays(self):
        self._cwghts = putils.exportArrayToNumpyArray(self.cwghts)
        self._nwghts = putils.exportArrayToNumpyArray(self.nwghts)
        self._xpins = putils.exportArrayToNumpyArray(self.xpins)
        self._pins = putils.exportArrayToNumpyArray(self.pins)
        self._partvec = putils.exportArrayToNumpyArray(self.partvec)

        self._targetweights = putils.exportArrayToNumpyArray(self.targetweights, dtype=np.float32)
        self._partweights = putils.exportArrayToNumpyArray(self.partweights)

    def _hyperedgeExpansionComplete(self, hyperedge_edge_weights, hyperedgeExpansionMode):
        '''
            This method takes all the edge weights and generates a complete graph between nodes
            It then applies the normal _hyperedgeExpansion.

            E.g. if a hyperedge has nodes with weights A, B, C, D, then _hyperedgeExpansion with product_sqrt mode would compute:
            sqrt(A*B*C*D).

            With _hyperedgeExpansionComplete we compute instead:
            Step 1:
            (A*B)^2 = H0
            (A*C)^2 = H1
            (A*D)^2 = H2
            (B*C)^2 = H3
            (B*D)^2 = H4
            (C*D)^2 = H5
            Step 2:
            _hyperedgeExpansion([H0, H1, ..., H5])
        '''
        # generate complete graph of nodes for the number of hyperedge nodes
        g = nx.complete_graph(len(hyperedge_edge_weights))

        # store all the new hyperedge weights
        hWeightsNew = []
        for edge in g.edges():
            w1 = hyperedge_edge_weights[edge[0]]
            w2 = hyperedge_edge_weights[edge[1]]

            #h = (w1 * w2) ** 0.5
            if(w1 < w2):
                h = w1
            else:
                h = w2
            #h = (w1 * w2) ** 2

            hWeightsNew.append(h)

        #if(len(hWeightsNew) == 0):
        #    hWeightsNew = hyperedge_edge_weights

        return self._hyperedgeExpansion(hWeightsNew, hyperedgeExpansionMode)
        #print('Hyperedge?', hWeightsNew, expansion)
        #return expansion

    def _hyperedgeExpansion(self, hyperedge_edge_weights, hyperedgeExpansionMode):
        ''' Net is the clique (hyperedge) '''
        '''
        1. average node weights on hyperedge
        2. total node weights
        3. smallest node weights
        4. largest node weight
        5. product of the node weights
        6. see explanation
        7. square of the above or sqrt
        8. not add any extra calculation
        '''
        if 'no_expansion' in hyperedgeExpansionMode:
            return 1

        hyperedgeWeight = 0.0
        for i, edgeWeight in enumerate(hyperedge_edge_weights):
            if 'avg_node_weight' in hyperedgeExpansionMode:
                hyperedgeWeight += edgeWeight
                if ((i + 1) == len(hyperedge_edge_weights)):
                    # last item
                    hyperedgeWeight = hyperedgeWeight / len(hyperedge_edge_weights)
            elif 'total_node_weight' in hyperedgeExpansionMode:
                hyperedgeWeight += edgeWeight
            elif 'smallest_node_weight' in hyperedgeExpansionMode:
                if i == 0:
                    hyperedgeWeight = edgeWeight
                else:
                    if edgeWeight < hyperedgeWeight:
                        hyperedgeWeight = edgeWeight
            elif 'largest_node_weight' in hyperedgeExpansionMode:
                if i == 0:
                    hyperedgeWeight = edgeWeight
                else:
                    if edgeWeight > hyperedgeWeight:
                        hyperedgeWeight = edgeWeight
            elif 'product_node_weight' in hyperedgeExpansionMode:
                if i == 0:
                    hyperedgeWeight = 1.0
                hyperedgeWeight = hyperedgeWeight * edgeWeight

        if 'squared' in hyperedgeExpansionMode:
            # take the square
            hyperedgeWeight = hyperedgeWeight ** 2.0

        if 'sqrt' in hyperedgeExpansionMode:
            hyperedgeWeight = hyperedgeWeight ** 0.5

        rounded = round(hyperedgeWeight)
        if(rounded <= 0):
            rounded = 1
        return rounded
        #return round(hyperedgeWeight)
