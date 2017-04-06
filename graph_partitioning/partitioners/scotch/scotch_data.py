import numpy as np
import networkx as nx

import graph_partitioning.partitioners.utils as putils

class ScotchData:
    def __init__(self):
        self._initialize()

    def _initialize(self):
        self.verttab = []
        self.edgetab = []
        self.edlotab = []
        self.velotab = []
        self.vlbltab = []
        self.vertexweights = []
        self.parttab = []

        self._verttab = None
        self._edgetab = None
        self._edlotab = None
        self._velotab = None
        self._vlbltab = None
        self._vertexweights = None
        self._parttab = None


        self.vertnbr = 0
        self.edgenbr = 0
        self.baseval = 0

    def debugPrint(self):
        print('vertnbr', self.vertnbr)
        print('edgenbr', self.edgenbr)
        print('baseval', self.baseval)

        print('len verttab', len(self.verttab))
        print('verttab', self.verttab)
        print('len velotab', len(self.velotab))
        print('velotab', self.velotab)
        print('len vlbltab', len(self.vlbltab))
        print('vlbltab', self.vlbltab)
        #print('len vertweights', len(self.vertexweights))
        #print('vertweights', self.vertexweights)

        print('len edgetab', len(self.edgetab))
        print('edgetab', self.edgetab)
        print('len edlotab', len(self.edlotab))
        print('edlotab', self.edlotab)

        print('len parttab', len(self.parttab))
        print('parttab', self.parttab)

    def isValid(self):
        # TODO complete this
        if self.vertnbr + 1 != len(self._verttab):
            return False
        if self.vertnbr != len(self._velotab):
            return False
        if self.edgenbr != len(self._edgetab):
            return False
        if self.edgenbr != len(self._edlotab):
            return False

        # deep check
        for edgeID in self._edgetab:
            if edgeID not in self.vlbltab:
                print('EdgeID not in vlbltab', edgeID)
                return False

        return True


    def clearData(self):
        self._initialize()


    def fromNetworkxGraph(self, nxGraph, baseval=1, parttab=None, vlbltab=None):
        if isinstance(nxGraph, nx.Graph) == False:
            print('Error, cannot load networkx graph from datatype', type(metisGraph).__name__)
            return False

        # number_of_nodes
        # size() ? number of edges

        self.vertnbr = nxGraph.number_of_nodes()
        self.edgenbr = nxGraph.size() * 2
        self.baseval = baseval

        self.verttab = putils.genArray(nxGraph.number_of_nodes() + 1)
        self.edgetab = putils.genArray(nxGraph.size() * 2)

        self.edlotab = putils.genArray(nxGraph.size() * 2)
        self.velotab = putils.genArray(nxGraph.number_of_nodes())


        if(vlbltab is None):
            self.vlbltab = putils.genArray(nxGraph.number_of_nodes())
            #self.vlbltab = []
        else:
            if len(vlbltab) == self.vertnbr:
                self.vlbltab = vlbltab
            else:
                self.vlbltab = putils.genArray(nxGraph.number_of_nodes())


        if parttab is None:
            self.parttab = putils.genArray(nxGraph.number_of_nodes(), -1)
        else:
            if len(parttab) == self.vertnbr:
                self.parttab = parttab
            else:
                self.parttab = putils.genArray(nxGraph.number_of_nodes(), -1)

        vtabID = 0
        nodes = sorted(nxGraph.nodes())

        vertCount = 0
        for vertexID in range(self.baseval, len(nodes) + self.baseval):
            vertex = nodes[vertexID - self.baseval]
            adjustedID = vertexID - self.baseval

            self.vlbltab[vertCount] = nodes[vertexID - self.baseval] # store the lable for this vertex as vertCount != adjustID
            vertCount += 1
            #vertex.printData(False)

            self.verttab[adjustedID] = vtabID

            vWeight = 1

            try:
                vWeight = int(nxGraph.node[vertex]['weight'])
            except KeyError as ke:
                pass

            self.velotab[adjustedID] = vWeight

            indexedEdges = {}
            edgeIndeces = sorted(nxGraph.neighbors(vertex))

            edgeCount = 0
            for edgeID in edgeIndeces:

                edgeWeight = 1
                try:
                    edgeWeight = int(nxGraph.edge[adjustedID][edgeID]['weight'])
                except Exception as e:
                    edgeWeight = 1

                self.edgetab[vtabID + edgeCount] = edgeID - self.baseval
                self.edlotab[vtabID + edgeCount] = edgeWeight

                #print('edge:', vertex, edgeID - self.baseval)

                edgeCount += 1
            vtabID += len(edgeIndeces)

        self.verttab[nxGraph.number_of_nodes()] = vtabID

        # update vertex IDs
        updateEdgeIDSUsingLabels = False
        if updateEdgeIDSUsingLabels:
            lblmap = {}
            for newVertID in range(0, len(self.vlbltab)):
                oldVertID = self.vlbltab[newVertID]
                lblmap[oldVertID] = newVertID
            for i in range(0, len(self.edgetab)):
                newVal = lblmap[self.edgetab[i]]
                self.edgetab[i] = newVal

        self._exportArrays()

    def setFixedVertices(self, parttab):
        if(len(parttab) == self.vertnbr):
            self.parttab = parttab
            self._parttab = putils.exportArrayToNumpyArray(parttab)
            return True
        return False

    def _exportArrays(self):
        self._verttab = putils.exportArrayToNumpyArray(self.verttab)
        self._edgetab = putils.exportArrayToNumpyArray(self.edgetab)
        self._edlotab = putils.exportArrayToNumpyArray(self.edlotab)
        self._velotab = putils.exportArrayToNumpyArray(self.velotab)
        self._parttab = putils.exportArrayToNumpyArray(self.parttab)
        self._vertexweights = putils.exportArrayToNumpyArray(self.vertexweights)
        if(len(self.vlbltab) == self.vertnbr):
            self._vlbltab = putils.exportArrayToNumpyArray(self.vlbltab)
