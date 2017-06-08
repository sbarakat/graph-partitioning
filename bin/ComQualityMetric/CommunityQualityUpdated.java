import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class CommunityQualityUpdated {
	public static void main(String[] args) {

		boolean isUnweighted = true;
		boolean isUndirected = true;
		String networkFile = "./football.txt";
		String discoveredCommunityFile = "./football_disjoint_community.groups";
		String groundTruthCommunityFile = "";

		int i = 0;
		String arg;
		while (i < args.length && args[i].startsWith("-")) {
			arg = args[i++];
			if (arg.equals("-weighted"))
				isUnweighted = false;
			if (arg.equals("-directed"))
				isUndirected = false;
		}
		if (args.length < i+2) {
			System.err.println("Usage: CommunityQualityUpdated [-weighted] [-directed] <networkFile> <discoveredCommunityFile> [groundTruthCommunityFile]");
			System.exit(1);
		}

		networkFile = args[i++];
		discoveredCommunityFile = args[i++];
		if (args.length == i+1) {
			groundTruthCommunityFile = args[i++];
		}

		// Calculate community quality metrics without ground truth
		double[] qualities = CommunityQuality.computeQualityWithoutGroundTruth(
				networkFile, isUnweighted, isUndirected,
				discoveredCommunityFile);
		System.out.println("Q = " + qualities[0] + ", Qds = " + qualities[1]
				+ ", intraEdges = " + qualities[2] + ", intraDensity = "
				+ qualities[3] + ", contraction = " + qualities[4]
				+ ", interEdges = " + qualities[5] + ", expansion = "
				+ qualities[6] + ", conductance = " + qualities[7]
				+ ", fitness = " + qualities[8] + ", modularity degree = "
				+ qualities[9]);

		if(groundTruthCommunityFile.length() > 0) {
			// Calculate seven metrics based on ground truth communities
			qualities = CommunityQuality.computeInformationEntropyMetrics(
					discoveredCommunityFile, groundTruthCommunityFile);
			System.out.println("VI = " + qualities[0] + ", NMI = " + qualities[1]);

			qualities = CommunityQuality.computeClusterMatchingMetrics(
					discoveredCommunityFile, groundTruthCommunityFile);
			System.out.println("F-measure = " + qualities[0] + ", NVD = "
					+ qualities[1]);

			qualities = CommunityQuality.computeIndexMetrics(
					discoveredCommunityFile, groundTruthCommunityFile);
			System.out.println("RI = " + qualities[0] + ", ARI = " + qualities[1]
					+ ", JI = " + qualities[2]);
		}
	}

	/**
	 * Calculate all the community quality metrics (including local and global
	 * ones) without ground truth communities
	 *
	 * @param networkFile
	 * @param isUnweighted
	 * @param isUndirected
	 * @param communityFile
	 * @return
	 */
	public static double[] computeQualityWithoutGroundTruth(String networkFile,
			boolean isUnweighted, boolean isUndirected, String communityFile) {
		// Get outgoing network
		HashMap<Integer, HashMap<Integer, Double>> outNet = new HashMap<Integer, HashMap<Integer, Double>>();
		// Return total weight. if undirected: 2m; if directed: m
		double[] weights = CommunityQuality.getNetwork(networkFile,
				isUnweighted, isUndirected, outNet);
		double totalWeight = weights[0];

		// Get communities and their nodes
		Map<Integer, Set<Integer>> communities = CommunityQuality
				.getMapCommunities(communityFile);
		int numComs = communities.size();
		double[][] communityWeights = new double[numComs][numComs];
		double[][] communityDensities = new double[numComs][numComs];

		// Get the community of each node
		Map<Integer, Integer> nodeCommunities = CommunityQuality
				.getNodeCommunities(communityFile);

		for (Map.Entry<Integer, HashMap<Integer, Double>> nodeItem : outNet
				.entrySet()) {
			int nodeId = nodeItem.getKey();
			int comId = nodeCommunities.get(nodeId);
			long comSize = communities.get(comId).size();
			HashMap<Integer, Double> nodeNbs = nodeItem.getValue();

			for (Map.Entry<Integer, Double> nbItem : nodeNbs.entrySet()) {
				int nbNodeId = nbItem.getKey();
				int nbComId = nodeCommunities.get(nbNodeId);
				long nbComSize = communities.get(nbComId).size();
				double weight = nbItem.getValue();
				communityWeights[comId][nbComId] += weight;
				if (comId == nbComId) {
					communityDensities[comId][comId] += 1.0 / (comSize * (comSize - 1));

					if (comSize <= 1) {
						// Used to debug
						System.out.println("comSize = " + comSize);
					}
				} else {
					communityDensities[comId][nbComId] += 1.0 / (comSize * nbComSize);
				}
			}
		}

		// Modularity
		double Q = 0;
		// Modularity Density
		double Qds = 0;

		// Intra-edges
		double intraEdges = 0;
		// Intra-density
		double intraDensity = 0;
		// Contraction
		double contraction = 0;
		// Inter-edges
		double interEdges = 0;
		// Expansion
		double expansion = 0;
		// Condunctance
		double conductance = 0;
		// Fitness function
		double fitness = 0;

		// Average modularity degree metric
		double D = 0;

		// The number of edges (including weight) inside a community
		double win = 0;
		double inDensity = 0;
		// The outgoing edges (including weight) from nodes inside the community
		// to nodes outside the community
		double wout = 0;
		// The incoming edges from nodes outside the community to nodes inside
		// the community
		double wout_incoming = 0;
		// Split penalty of the community
		double sp = 0;

		for (int i = 0; i < numComs; ++i) {
			Set<Integer> community = communities.get(i);
			long iCommSize = community.size();
			win = communityWeights[i][i];
			inDensity = communityDensities[i][i];
			wout = 0;
			wout_incoming = 0;
			sp = 0;
			for (int j = 0; j < numComs; ++j) {
				if (j != i) {
					double wij = communityWeights[i][j];
					double wji = communityWeights[j][i];
					sp += wij * communityDensities[i][j];
					wout += wij;
					wout_incoming += wji;
				}
			}

			if (isUndirected) {
				Q += win / totalWeight
						- Math.pow((win + wout) / totalWeight, 2);
				Qds += (win / totalWeight) * inDensity
						- Math.pow(((win + wout) / totalWeight) * inDensity, 2)
						- sp / totalWeight;
			} else {
				Q += win / totalWeight - ((win + wout) * (win + wout_incoming))
						/ Math.pow(totalWeight, 2);
				Qds += (win / totalWeight)
						* inDensity
						- (((win + wout) * (win + wout_incoming)) / (totalWeight * totalWeight))
						* Math.pow(inDensity, 2) - sp / totalWeight;
			}

			// intra-edges
			if (isUndirected) {
				intraEdges += win / 2;
			} else {
				intraEdges += win;
			}

			// contraction: average degree
			contraction += win / iCommSize;

			// intra-density
			if (iCommSize == 1) {
				intraDensity += 0;
			} else {
				intraDensity += inDensity;
			}

			interEdges += wout;
			expansion += wout / iCommSize;

			// Conductance
			if (wout == 0) {
				conductance += 0;
			} else {
				conductance += wout / (win + wout);
			}

			// The fitness function
			if (win == 0) {
				fitness += 0;
			} else {
				if (isUndirected) {
					fitness += win / (win + 2 * wout);
				} else {
					fitness += win / (win + wout + wout_incoming);
				}
			}

			// Average modularity degree
			D += (win - wout) / iCommSize;
		}

		double[] qualities = { Q, Qds, intraEdges / numComs,
				intraDensity / numComs, contraction / numComs,
				interEdges / numComs, expansion / numComs,
				conductance / numComs, fitness / numComs, D };
		return qualities;
	}

	/**
	 * Compute Variation Information (VI) and Normalized Mutual Information
	 * (NMI): Disjoint community quality
	 *
	 * @param discoveredCommunities
	 * @param realCommunities
	 * @return
	 */
	public static double[] computeInformationEntropyMetrics(
			String disCommunityFile, String realCommunityFile) {
		List<Set<Integer>> discoveredCommunities = CommunityQuality
				.getCommunities(disCommunityFile);
		List<Set<Integer>> realCommunities = CommunityQuality
				.getCommunities(realCommunityFile);

		// H(X)
		double xentropy = 0;
		// H(Y)
		double yentropy = 0;
		// VI(X, Y)
		double variationInformation = 0;
		// I(X, Y)
		double nmi = 0;
		int realsize = realCommunities.size();
		int exsize = discoveredCommunities.size();
		// total number of nodes
		double numNodes = 0;
		Set<Integer> nodes = new HashSet<Integer>();

		for (int i = 0; i < realsize; ++i) {
			Set<Integer> realCommunity = realCommunities.get(i);
			// double ni = realCommunity.size();
			// numNodes += ni;
			nodes.addAll(realCommunity);
		}
		numNodes = nodes.size();

		// System.out.println("numNodes = " + numNodes + ", realSize = "
		// + realsize + ", exsize = " + exsize);

		for (int i = 0; i < realsize; ++i) {
			Set<Integer> realCommunity = realCommunities.get(i);
			double ni = realCommunity.size();
			xentropy += -(ni / numNodes)
					* (Math.log10(ni / numNodes) / Math.log10(2));
		}

		for (int j = 0; j < exsize; ++j) {
			Set<Integer> exCommunity = discoveredCommunities.get(j);
			double nj = exCommunity.size();
			yentropy += -(nj / numNodes)
					* (Math.log10(nj / numNodes) / Math.log10(2));
		}

		for (int i = 0; i < realsize; ++i) {
			Set<Integer> realCommunity = realCommunities.get(i);
			double ni = realCommunity.size();
			for (int j = 0; j < exsize; ++j) {
				Set<Integer> exCommunity = discoveredCommunities.get(j);
				double nj = exCommunity.size();
				double nij = CommunityQuality.getIntersectionNumber(
						realCommunity, exCommunity);
				// log2
				if (nij != 0) {
					variationInformation += nij
							* (Math.log10((nij * nij) / (ni * nj)) / Math
									.log10(2));
					nmi += (nij / numNodes)
							* (Math.log10((nij * numNodes) / (ni * nj)) / Math
									.log10(2));
				}
			}
		}

		variationInformation = variationInformation / (-numNodes);
		if (nmi == 0) {
			nmi = 0;
			if (realsize == 1 && exsize == 1) {
				nmi = 1;
			}
		} else {
			// System.out.println("xentropy=" + xentropy + ", yentropy="
			// + yentropy + ", 2*nmi=" + (2 * nmi));
			nmi = (2 * nmi) / (xentropy + yentropy);
		}
		double[] entropyMetrics = { variationInformation, nmi };
		return entropyMetrics;
	}

	/**
	 * Compute F-measure and Normalized Van Dongen metric (NVD): Disjoint
	 * community quality
	 *
	 * @param disCommunityFile
	 * @param realCommunityFile
	 * @return
	 */
	public static double[] computeClusterMatchingMetrics(
			String disCommunityFile, String realCommunityFile) {
		List<Set<Integer>> discoveredCommunities = CommunityQuality
				.getCommunities(disCommunityFile);
		List<Set<Integer>> realCommunities = CommunityQuality
				.getCommunities(realCommunityFile);

		double fMeasure = 0;
		double nvd = 0;
		int realsize = realCommunities.size();
		int exsize = discoveredCommunities.size();
		// total number of nodes
		double numNodes = 0;
		Set<Integer> nodes = new HashSet<Integer>();

		for (int j = 0; j < realsize; ++j) {
			Set<Integer> realCommunity = realCommunities.get(j);
			// double nj = realCommunity.size();
			// numNodes += nj;
			nodes.addAll(realCommunity);
		}
		numNodes = nodes.size();

		for (int i = 0; i < exsize; ++i) {
			Set<Integer> exCommunity = discoveredCommunities.get(i);
			double maxCommon = -1;

			for (int j = 0; j < realsize; ++j) {
				Set<Integer> realCommunity = realCommunities.get(j);
				double nij = CommunityQuality.getIntersectionNumber(
						exCommunity, realCommunity);
				if (nij > maxCommon) {
					maxCommon = nij;
				}
			}

			nvd += maxCommon;
		}

		for (int j = 0; j < realsize; ++j) {
			Set<Integer> realCommunity = realCommunities.get(j);
			double nj = realCommunity.size();
			double maxCommon = -1;
			double maxFMeasure = Double.NEGATIVE_INFINITY;

			for (int i = 0; i < exsize; ++i) {
				Set<Integer> exCommunity = discoveredCommunities.get(i);
				double ni = exCommunity.size();
				double nji = CommunityQuality.getIntersectionNumber(
						realCommunity, exCommunity);

				if (nji > maxCommon) {
					maxCommon = nji;
				}

				double tmpFMeasure = (2 * nji) / (ni + nj);
				if (tmpFMeasure > maxFMeasure) {
					maxFMeasure = tmpFMeasure;
				}
			}

			nvd += maxCommon;
			fMeasure += nj * maxFMeasure;
		}

		fMeasure = fMeasure / numNodes;
		nvd = 1 - nvd / (2 * numNodes);

		double[] cmMetrics = { fMeasure, nvd };
		return cmMetrics;
	}

	/**
	 * Compute Rand Index and Jaccard Index: Disjoint community quality
	 *
	 * @param disCommunityFile
	 * @param realCommunityFile
	 * @return
	 */
	public static double[] computeIndexMetrics(String disCommunityFile,
			String realCommunityFile) {
		double randIndex = 0;
		double jaccardIndex = 0;

		Map<Integer, Integer> disNodeCommunities = CommunityQuality
				.getNodeCommunities(disCommunityFile);
		Map<Integer, Integer> realCommunities = CommunityQuality
				.getNodeCommunities(realCommunityFile);
		ArrayList<Integer> nodes = new ArrayList<Integer>(
				disNodeCommunities.keySet());
		int numNodes = nodes.size();
		// System.out.println(numNodes + ", " + disNodeCommunities.size());
		// System.out.println(nodes.toString());
		double a11 = 0;
		double a00 = 0;
		double a10 = 0;
		double a01 = 0;
		for (int i = 0; i < numNodes; ++i) {
			int inode = nodes.get(i);
			for (int j = i + 1; j < numNodes; ++j) {
				int jnode = nodes.get(j);
				if (inode != jnode) {
					int icom = disNodeCommunities.get(inode);
					int jcom = disNodeCommunities.get(jnode);
					boolean dflag = true;
					if (icom != jcom) {
						dflag = false;
					}
					icom = realCommunities.get(inode);
					jcom = realCommunities.get(jnode);
					boolean rflag = true;
					if (icom != jcom) {
						rflag = false;
					}

					if (dflag && rflag) {
						a11 += 1;
					} else if (rflag && !dflag) {
						a10 += 1;
					} else if (!rflag && dflag) {
						a01 += 1;
					} else if (!rflag && !dflag) {
						a00 += 1;
					}
				}
			}
		} // for

		// System.out.println(a11 + ", " + a10 + ", " + a01 + ", " + a00);
		randIndex = (a11 + a00) / (a11 + a10 + a01 + a00);
		double m = a11 + a10 + a01 + a00;
		double adjustRandIndex = (a11 - ((a11 + a10) * (a11 + a01)) / m)
				/ ((a11 + a10 + a11 + a01) / 2 - ((a11 + a10) * (a11 + a01))
						/ m);
		jaccardIndex = a11 / (a11 + a10 + a01);
		// System.out.println("numNodes = " + numNodes);
		// System.out.println("a00=" + a00 + ", a10=" + a10 + ", a01=" + a01
		// + ", a11=" + a11);
		double[] indices = { randIndex, adjustRandIndex, jaccardIndex };
		return indices;
	}

	/**
	 * Get the intersection of 2 sets
	 *
	 * @param scomm
	 * @param gcomm
	 * @return
	 */
	public static int getIntersectionNumber(Set<Integer> scomm,
			Set<Integer> gcomm) {
		int num = 0;

		if (scomm == null || gcomm == null) {
			System.out.println("scomm or gcomm == null");
			return num;
		}

		int scommSize = scomm.size();
		int gcommSize = gcomm.size();
		Set<Integer> tmp1 = null, tmp2 = null;
		if (scommSize < gcommSize) {
			tmp1 = scomm;
			tmp2 = gcomm;
		} else {
			tmp1 = gcomm;
			tmp2 = scomm;
		}

		for (int nodeId : tmp1) {
			if (tmp2.contains(nodeId)) {
				++num;
			}
		}

		return num;
	}

	/**
	 * Get the union of 2 sets
	 *
	 * @param scomm
	 * @param gcomm
	 * @return
	 */
	public static int getUnionNumber(Set<Integer> scomm, Set<Integer> gcomm) {
		if (scomm == null) {
			System.out.println("scomm == null");
			return gcomm.size();
		}

		if (gcomm == null) {
			System.out.println("gcomm == null");
			return scomm.size();
		}

		Set<Integer> tempSet = new HashSet<Integer>();
		tempSet.addAll(scomm);
		tempSet.addAll(gcomm);
		return tempSet.size();
	}

	/**
	 * The id of communities start from 0
	 *
	 * @param communityFile
	 * @return
	 */
	public static Map<Integer, Integer> getNodeCommunities(String communityFile) {
		Map<Integer, Integer> nodeCommunities = new HashMap<Integer, Integer>();
		try {
			BufferedReader br = new BufferedReader(new InputStreamReader(
					new FileInputStream(communityFile)));
			String tmp = null;
			String[] nodes = null;
			// The id of communities start from 0
			int count = 0;

			while ((tmp = br.readLine()) != null) {
				tmp = tmp.trim();
				if (tmp.charAt(0) == '#') {
					continue;
				}
				tmp = tmp.replaceAll("\\s+", " ");
				nodes = tmp.split(" ");

				for (int i = 0; i < nodes.length; ++i) {
					nodeCommunities.put(Integer.parseInt(nodes[i]), count);
				}

				++count;
			}

			br.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}

		return nodeCommunities;
	}

	public static List<Set<Integer>> getCommunities(String communityFile) {
		List<Set<Integer>> communities = new ArrayList<Set<Integer>>();
		try {
			BufferedReader br = new BufferedReader(new InputStreamReader(
					new FileInputStream(communityFile)));
			String tmp = null;
			String[] nodes = null;

			while ((tmp = br.readLine()) != null) {
				tmp = tmp.trim();
				if (tmp.charAt(0) == '#') {
					continue;
				}
				tmp = tmp.replaceAll("\\s+", " ");
				nodes = tmp.split(" ");
				Set<Integer> community = new HashSet<Integer>();
				for (int i = 0; i < nodes.length; ++i) {
					community.add(Integer.parseInt(nodes[i].trim()));
				}

				communities.add(community);
			}

			br.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}

		return communities;
	}

	/**
	 * The id of communities start from 0
	 *
	 * @param communityFile
	 * @return
	 */
	public static Map<Integer, Set<Integer>> getMapCommunities(
			String communityFile) {
		Map<Integer, Set<Integer>> community = new HashMap<>();
		try {
			BufferedReader br = new BufferedReader(new InputStreamReader(
					new FileInputStream(communityFile)));
			String tmp = null;
			String[] values = null;
			// The id of communities start from 0
			int count = 0;

			while ((tmp = br.readLine()) != null) {
				tmp = tmp.trim();
				if (tmp.charAt(0) == '#') {
					continue;
				}
				tmp = tmp.replaceAll("\\s+", " ");
				values = tmp.split(" ");
				int length = values.length;

				if (!community.containsKey(count)) {
					community.put(count, new HashSet<Integer>());
				}

				Set<Integer> communityNodes = community.get(count);

				for (int i = 0; i < length; ++i) {
					communityNodes.add(Integer.parseInt(values[i]));
				}

				++count;
			}

			br.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}

		return community;
	}

	/**
	 * Get the network and return totalWeight. totalWeight=2m for undirected
	 * network; totalWeight=m for directed network. If the network is weighted,
	 * then including weights of edges
	 *
	 * @param fileName
	 * @param isUnweighted
	 * @param isUndirected
	 * @param net
	 * @return
	 */
	public static double[] getNetwork(String fileName, boolean isUnweighted,
			boolean isUndirected, HashMap<Integer, HashMap<Integer, Double>> net) {
		double totalWeight = 0;
		// return the max weight of the network
		double maxWeight = Double.NEGATIVE_INFINITY;
		try {
			BufferedReader br = new BufferedReader(new FileReader(fileName));
			String tmp = null;
			String values[] = null;
			int srcId = 0;
			int dstId = 0;
			double weight = 0;

			while ((tmp = br.readLine()) != null) {
				tmp = tmp.trim();
				tmp = tmp.replaceAll("\\s+", " ");
				values = tmp.split(" ");
				srcId = Integer.parseInt(values[0]);
				dstId = Integer.parseInt(values[1]);

				// delete self-loop edges
				if (srcId == dstId) {
					// System.out.println("self-loop");
					continue;
				}

				weight = 1;
				if (!isUnweighted) {
					weight = Double.parseDouble(values[2]);
				}

				if (weight > maxWeight) {
					maxWeight = weight;
				}

				if (!net.containsKey(srcId)) {
					net.put(srcId, new HashMap<Integer, Double>());
				}

				HashMap<Integer, Double> neighbors = net.get(srcId);
				if (!neighbors.containsKey(dstId)) {
					neighbors.put(dstId, weight);
					totalWeight += weight;
				}

				if (!net.containsKey(dstId)) {
					net.put(dstId, new HashMap<Integer, Double>());
				}

				if (isUndirected) {
					neighbors = net.get(dstId);
					if (!neighbors.containsKey(srcId)) {
						neighbors.put(srcId, weight);
						totalWeight += weight;
					}
				}
			}

			br.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}

		// System.out.println("Finish read the network");
		double[] weights = { totalWeight, maxWeight };
		return weights;
	}

	/**
	 * Get the network with edge direction reversed
	 *
	 * @param fileName
	 * @param isUnweighted
	 * @param isUndirected
	 * @param net
	 */
	public static void getReversedNetwork(String fileName,
			boolean isUnweighted, boolean isUndirected,
			HashMap<Integer, HashMap<Integer, Double>> net) {
		try {
			BufferedReader br = new BufferedReader(new FileReader(fileName));
			String tmp = null;
			String values[] = null;
			int srcId = 0;
			int dstId = 0;
			double weight = 0;

			while ((tmp = br.readLine()) != null) {
				tmp = tmp.trim();
				tmp = tmp.replaceAll("\\s+", " ");
				values = tmp.split(" ");
				srcId = Integer.parseInt(values[0]);
				dstId = Integer.parseInt(values[1]);

				// delete self-loop edges
				if (srcId == dstId) {
					continue;
				}

				weight = 1;
				if (!isUnweighted) {
					weight = Double.parseDouble(values[2]);
				}

				if (!net.containsKey(dstId)) {
					net.put(dstId, new HashMap<Integer, Double>());
				}

				HashMap<Integer, Double> neighbors = net.get(dstId);
				if (!neighbors.containsKey(srcId)) {
					neighbors.put(srcId, weight);
				}

				if (!net.containsKey(srcId)) {
					net.put(srcId, new HashMap<Integer, Double>());
				}

				if (isUndirected) {
					neighbors = net.get(srcId);
					if (!neighbors.containsKey(dstId)) {
						neighbors.put(dstId, weight);
					}
				}
			}

			br.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
