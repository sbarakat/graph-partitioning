import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class OverlappingCommunityQuality {

	/**
	 * @param args
	 */
	public static void main(String[] args) {

		int i = 0;
		String arg;
		boolean isUnweighted = true;
		boolean isUndirected = true;
		String networkFile = "";
		String discoveredCommunityFile = "";
		String groundTruthCommunityFile = "";

		while (i < args.length && args[i].startsWith("-")) {
			arg = args[i++];
			if (arg.equals("-weighted"))
				isUnweighted = false;
			if (arg.equals("-directed"))
				isUndirected = false;
		}
		if (args.length < i+2) {
			System.err.println("Usage: OverlappingCommunityQuality [-weighted] [-directed] <networkFile> <discoveredCommunityFile> [groundTruthCommunityFile]");
			System.exit(1);
		}
		networkFile = args[i++];
		discoveredCommunityFile = args[i++];
		if (args.length == i+1) {
			groundTruthCommunityFile = args[i++];
		}

		// Please look at the function definition for the value of
		// belongingVersion and belongingFunVersion
		int belongingVersion = 1;
		int belongingFunVersion = 1;
		double[] qualities = OverlappingCommunityQuality
				.computeOvQualityWithoutGroundTruth(networkFile, isUnweighted,
						isUndirected, discoveredCommunityFile,
						belongingVersion, belongingFunVersion);
		System.out.println("Q = " + qualities[0] + ", NQ = " + qualities[1]
				+ ", Qds = " + qualities[2] + ", intraEdges = " + qualities[3]
				+ ", intraDensity = " + qualities[4] + ", contraction = "
				+ qualities[5] + ", interEdges = " + qualities[6]
				+ ", expansion = " + qualities[7] + ", conductance = "
				+ qualities[8] + ", fitness = " + qualities[9]
				+ ", modularity degree = " + qualities[10]);

		// The same as computeOvQualityWithoutGroundTruth(), except using
		// different data structure
		// qualities = OverlappingCommunityQuality
		// .computeOvQualityWithoutGroundTruthWithMap(networkFile,
		// isUnweighted, isUndirected, discoveredCommunityFile,
		// belongingVersion, belongingFunVersion);
		// System.out.println("Q = " + qualities[0] + ", NQ = " + qualities[1]
		// + ", Qds = " + qualities[2] + ", intraEdges = " + qualities[3]
		// + ", intraDensity = " + qualities[4] + ", contraction = "
		// + qualities[5] + ", interEdges = " + qualities[6]
		// + ", expansion = " + qualities[7] + ", conductance = "
		// + qualities[8] + ", fitness = " + qualities[9]
		// + ", modularity degree = " + qualities[10]);

		double QovLink = OverlappingCommunityQuality.computeQovLink(
				networkFile, isUnweighted, isUndirected,
				discoveredCommunityFile, belongingVersion, belongingFunVersion);
		System.out.println("QovL = " + QovLink);

		if (!groundTruthCommunityFile.isEmpty()) {
			double NMI = OverlappingCommunityQuality.computeNMI(
					discoveredCommunityFile, groundTruthCommunityFile);
			System.out.println("NMI = " + NMI);

			double omega = OverlappingCommunityQuality.computeOmegaIndex(
					discoveredCommunityFile, groundTruthCommunityFile);
			System.out.println("Omega = " + omega);

			double[] fscore = OverlappingCommunityQuality.computeFscore(
					discoveredCommunityFile, groundTruthCommunityFile);
			System.out.println("Precision = " + fscore[0] + ", recall = "
					+ fscore[1] + ", Fscore = " + fscore[2]);

			double[] f1 = OverlappingCommunityQuality.computeF1(
					discoveredCommunityFile, groundTruthCommunityFile);
			System.out.println("Precision = " + f1[0] + ", recall = " + f1[1]
					+ ", F1 = " + f1[2]);
		}
	}

	/**
	 * All the Overlapping community quality metrics (including local and
	 * global) without ground truth communities, with belonging coefficient and
	 * belonging function
	 * 
	 * @param networkFile
	 * @param isUnweighted
	 * @param isUndirected
	 * @param communityFile
	 * @param belongingVersion
	 *            0: fuzzy overlapping; 1: crisp overlapping with belonging
	 *            coefficients being 1/O_i; 2: crisp overlapping with belonging
	 *            coefficients being the strength of the node to the community.
	 * @param belongingFunVersion
	 *            0: average; 1: product; 2: max.
	 * @return
	 */
	public static double[] computeOvQualityWithoutGroundTruth(
			String networkFile, boolean isUnweighted, boolean isUndirected,
			String communityFile, int belongingVersion, int belongingFunVersion) {
		// long startTime = System.currentTimeMillis();

		// Get outgoing network
		HashMap<Integer, HashMap<Integer, Double>> outNet = new HashMap<Integer, HashMap<Integer, Double>>();
		// Return total weight. if undirected: 2m; if directed: m
		double[] weights = CommunityQuality.getNetwork(networkFile,
				isUnweighted, isUndirected, outNet);
		double totalWeight = weights[0];
		// double maxWeight = weights[1];
		// long numNodes = outNet.size();

		// System.out.println("#node = " + numNodes);

		// If network is directed, get incoming network
		HashMap<Integer, HashMap<Integer, Double>> inNet = null;
		if (!isUndirected) {
			inNet = new HashMap<Integer, HashMap<Integer, Double>>();
			CommunityQuality.getReversedNetwork(networkFile, isUnweighted,
					isUndirected, inNet);
		}

		// System.out.println("Finish reading the network.");
		Map<Integer, Set<Integer>> mapCommunities = CommunityQuality
				.getMapCommunities(communityFile);
		int numComs = mapCommunities.size();
		// System.out.println("#com = " + numComs);

		HashMap<Integer, HashMap<Integer, Double>> ovNodeCommunities = OverlappingCommunityQuality
				.getCrispOverlappingNodeCommunities(communityFile);

		if (belongingVersion == 1) {
			// Crisp overlapping with belonging coefficients being 1/O_i
			OverlappingCommunityQuality
					.convertCrispToFuzzyOvCommunityWithNumComs(ovNodeCommunities);
		} else if (belongingVersion == 2) {
			// Crisp overlapping with belonging coefficients being the strength
			// of the node to the community
			OverlappingCommunityQuality
					.convertCrispToFuzzyOvCommunityWithNodeStrength(outNet,
							inNet, isUndirected, mapCommunities,
							ovNodeCommunities);
		}

		// Use type float to save memory
		double[] communitySizes = new double[numComs];
		double[][] communityWeights = new double[numComs][numComs];
		double[][] communityDensities = new double[numComs][numComs];

		// ///////////////////////////////////////////
		for (Map.Entry<Integer, HashMap<Integer, Double>> nodeItem : outNet
				.entrySet()) {
			int nodeId = nodeItem.getKey();
			HashMap<Integer, Double> nodeNbs = nodeItem.getValue();
			HashMap<Integer, Double> nodeComs = ovNodeCommunities.get(nodeId);

			// Traverse the neighboring nodes of this node
			for (Map.Entry<Integer, Double> nbItem : nodeNbs.entrySet()) {
				int nbId = nbItem.getKey();
				double weight = nbItem.getValue();
				HashMap<Integer, Double> nbComs = ovNodeCommunities.get(nbId);

				for (Map.Entry<Integer, Double> nodeComItem : nodeComs
						.entrySet()) {
					int nodeComId = nodeComItem.getKey();
					double nodeComFactor = nodeComItem.getValue();
					for (Map.Entry<Integer, Double> nbComItem : nbComs
							.entrySet()) {
						int nbComId = nbComItem.getKey();
						double nbComFactor = nbComItem.getValue();

						// Depends on belongingFunVersion
						if (belongingFunVersion == 0) {
							communityDensities[nodeComId][nbComId] += (nodeComFactor + nbComFactor) / 2;
							communityWeights[nodeComId][nbComId] += weight
									* (nodeComFactor + nbComFactor) / 2;
						} else if (belongingFunVersion == 1) {
							communityDensities[nodeComId][nbComId] += nodeComFactor
									* nbComFactor;
							communityWeights[nodeComId][nbComId] += weight
									* nodeComFactor * nbComFactor;
						} else if (belongingFunVersion == 2) {
							communityDensities[nodeComId][nbComId] += Math.max(
									nodeComFactor, nbComFactor);
							communityWeights[nodeComId][nbComId] += weight
									* Math.max(nodeComFactor, nbComFactor);
						}
					}
				}
			} // node neighbors for
		} // nodes for
			// ///////////////////////////////////////

		for (int iComId = 0; iComId < numComs; ++iComId) {
			Set<Integer> iCommunity = mapCommunities.get(iComId);

			// Calculate community size with belonging factors
			for (int iNodeId : iCommunity) {
				communitySizes[iComId] += OverlappingCommunityQuality
						.getNodeBelongingFactor(ovNodeCommunities, iNodeId,
								iComId);
			}

			for (int jComId = 0; jComId < numComs; ++jComId) {

				// To save time, if there is no edge between the two
				// communities, we skip this pair
				if (communityWeights[iComId][jComId] == 0) {
					continue;
				}

				Set<Integer> jCommunity = mapCommunities.get(jComId);

				// Get the numerator and denominator of community density
				// double numerator = 0;
				double denominator = 0;
				for (int iNodeId : iCommunity) {
					// HashMap<Integer, Double> iNbs = outNet.get(iNodeId);
					double iFactor = OverlappingCommunityQuality
							.getNodeBelongingFactor(ovNodeCommunities, iNodeId,
									iComId);
					for (int jNodeId : jCommunity) {
						double jFactor = OverlappingCommunityQuality
								.getNodeBelongingFactor(ovNodeCommunities,
										jNodeId, jComId);
						if (iComId == jComId) {
							if (jNodeId != iNodeId) {
								if (belongingFunVersion == 0) {
									denominator += (iFactor + jFactor) / 2;
								} else if (belongingFunVersion == 1) {
									denominator += iFactor * jFactor;
								} else if (belongingFunVersion == 2) {
									denominator += Math.max(iFactor, jFactor);
								}
							}
						} else {
							if (belongingFunVersion == 0) {
								denominator += (iFactor + jFactor) / 2;
							} else if (belongingFunVersion == 1) {
								denominator += iFactor * jFactor;
							} else if (belongingFunVersion == 2) {
								denominator += Math.max(iFactor, jFactor);
							}
						}
					} // jNodeId for
				} // iNodeId for

				if (denominator == 0) {
					communityDensities[iComId][jComId] = 0;
				} else {
					communityDensities[iComId][jComId] = communityDensities[iComId][jComId]
							/ denominator;
				}
			} // j for
		} // i for

		// Compute average community quality of the sample
		double Q = 0;
		double NQ = 0;
		double Qds = 0;
		double intraEdges = 0;
		double intraDensity = 0;
		double contraction = 0;
		double interEdges = 0;
		// double interDensity = 0;
		double expansion = 0;
		double conductance = 0;
		// Fitness function
		double fitnessFunction = 0;

		// Average modularity degree metric
		double D = 0;

		for (int iComId = 0; iComId < numComs; ++iComId) {
			double iCommSize = communitySizes[iComId];
			double inWeights = communityWeights[iComId][iComId];
			double inDensity = communityDensities[iComId][iComId];
			// The number of outgoing edges from nodes inside the community to
			// the nodes ouside this community
			double outWeights = 0;
			// The number of incoming edges from nodes outside the community to
			// the nodes inside this community
			double out_incoming_weights = 0;
			double splitPenalty = 0;

			// The total weight of the subnetwork including the community and
			// its
			// neighboring communities
			double nebComTotalWeight = 0;
			HashSet<Integer> nebComs = new HashSet<Integer>();
			nebComs.add(iComId);

			for (int jComId = 0; jComId < numComs; ++jComId) {
				if (jComId != iComId) {
					double wij = communityWeights[iComId][jComId];
					double wji = communityWeights[jComId][iComId];
					outWeights += wij;
					if (wij != 0) {
						nebComs.add(jComId);
					}

					if (!isUndirected) {
						out_incoming_weights += wji;
						if (wji != 0) {
							nebComs.add(jComId);
						}
					}

					// double sp = (wij / totalWeight)
					// * communityDensities[iComId][jComId];
					double sp = wij * communityDensities[iComId][jComId];
					splitPenalty += sp;
				}
			}

			// Calculate nebComTotalWeight
			for (int icom : nebComs) {
				for (int jcom : nebComs) {
					nebComTotalWeight += communityWeights[icom][jcom];
				}
			}

			// System.out.println(inWeights + "\t" + outWeights + "\t"
			// + nebComTotalWeight + "\t" + nebNodeTotalWeight + "\t"
			// + wideNebNodeTotalWeight + "\t" + iCommSize + "\t"
			// + inDensity + "\t" + dc_out + "\t" + dc_out_denominator);

			if (isUndirected) {
				// modularity
				Q += inWeights / totalWeight
						- Math.pow((inWeights + outWeights) / totalWeight, 2);
				// Modularity Density Qds
				Qds += (inWeights / totalWeight)
						* inDensity
						- Math.pow(((inWeights + outWeights) / totalWeight)
								* inDensity, 2) - splitPenalty / totalWeight;

				if (nebComTotalWeight != 0) {
					NQ += inWeights
							/ nebComTotalWeight
							- Math.pow((inWeights + outWeights)
									/ nebComTotalWeight, 2);
				}
			} else {
				Q += inWeights
						/ totalWeight
						- ((inWeights + outWeights) * (inWeights + out_incoming_weights))
						/ Math.pow(totalWeight, 2);
				// Modularity Density Qds
				Qds += (inWeights / totalWeight)
						* inDensity
						- (((inWeights + outWeights) * (inWeights + out_incoming_weights)) / Math
								.pow(totalWeight, 2)) * Math.pow(inDensity, 2)
						- splitPenalty / totalWeight;

				if (nebComTotalWeight != 0) {
					NQ += inWeights
							/ nebComTotalWeight
							- ((inWeights + outWeights) * (inWeights + out_incoming_weights))
							/ Math.pow(nebComTotalWeight, 2);
				}
			}

			// intra-edges
			if (isUndirected) {
				intraEdges += inWeights / 2;
			} else {
				intraEdges += inWeights;
			}
			// contraction: average degree
			if (inWeights == 0 || iCommSize == 0) {
				contraction += 0;
			} else {
				contraction += inWeights / iCommSize;
			}
			// intra-density
			intraDensity += inDensity;
			interEdges += outWeights;
			// inter-density
			// if (numNodes == iCommSize) {
			// interDensity += 0;
			// } else {
			// interDensity += outWeights
			// / (iCommSize * (numNodes - iCommSize));
			// }
			if (outWeights == 0 || iCommSize == 0) {
				expansion += 0;
			} else {
				expansion += outWeights / iCommSize;
			}

			// Avoid that totalInterEdges==0 and communityEdges[i][i]==0
			if (outWeights == 0) {
				conductance += 0;
			} else {
				conductance += outWeights / (inWeights + outWeights);
			}

			// The fitness function
			if (inWeights == 0) {
				fitnessFunction += 0;
			} else {
				double fitness = 0;
				if (isUndirected) {
					fitness = inWeights / (inWeights + 2 * outWeights);
					fitnessFunction += fitness;
				} else {
					fitness = inWeights
							/ (inWeights + outWeights + out_incoming_weights);
					fitnessFunction += fitness;
				}
			}

			// Average modularity degree metric
			if (iCommSize == 0) {
				D += 0;
			} else {
				D += (inWeights - outWeights) / iCommSize;
			}
		} // for

		// for local community detection algorithms
		double[] qualities = { Q, NQ, Qds, intraEdges / numComs,
				intraDensity / numComs, contraction / numComs,
				interEdges / numComs, expansion / numComs,
				conductance / numComs, fitnessFunction / numComs, D };

		// long endTime = System.currentTimeMillis();
		// System.out.println("running time: " + (endTime - startTime) + "ms");

		return qualities;
	}

	/**
	 * All the Overlapping community quality metrics (including local and
	 * global) without ground truth communities, with belonging coefficient and
	 * belonging function
	 * 
	 * It is exactly the same with function computeOvQualityWithoutGroundTruth()
	 * above except it uses HashMap instead of array to save memory for sparse
	 * networks / communities
	 * 
	 * @param networkFile
	 * @param isUnweighted
	 * @param isUndirected
	 * @param communityFile
	 * @param belongingVersion
	 *            0: fuzzy overlapping; 1: crisp overlapping with belonging
	 *            coefficients being 1/O_i; 2: crisp overlapping with belonging
	 *            coefficients being the strength of the node to the community.
	 * @param belongingFunVersion
	 *            0: average; 1: product; 2: max.
	 * @return
	 */
	public static double[] computeOvQualityWithoutGroundTruthWithMap(
			String networkFile, boolean isUnweighted, boolean isUndirected,
			String communityFile, int belongingVersion, int belongingFunVersion) {
		// long startTime = System.currentTimeMillis();

		// Get outgoing network
		HashMap<Integer, HashMap<Integer, Double>> outNet = new HashMap<Integer, HashMap<Integer, Double>>();
		// Return total weight. if undirected: 2m; if directed: m
		double[] weights = CommunityQuality.getNetwork(networkFile,
				isUnweighted, isUndirected, outNet);
		double totalWeight = weights[0];
		// double maxWeight = weights[1];
		// long numNodes = outNet.size();

		// System.out.println("#node = " + numNodes);

		// If network is directed, get incoming network
		HashMap<Integer, HashMap<Integer, Double>> inNet = null;
		if (!isUndirected) {
			inNet = new HashMap<Integer, HashMap<Integer, Double>>();
			CommunityQuality.getReversedNetwork(networkFile, isUnweighted,
					isUndirected, inNet);
		}

		// System.out.println("Finish reading the network.");
		Map<Integer, Set<Integer>> mapCommunities = CommunityQuality
				.getMapCommunities(communityFile);
		int numComs = mapCommunities.size();
		// System.out.println("#com = " + numComs);

		HashMap<Integer, HashMap<Integer, Double>> ovNodeCommunities = OverlappingCommunityQuality
				.getCrispOverlappingNodeCommunities(communityFile);

		if (belongingVersion == 1) {
			// Crisp overlapping with belonging coefficients being 1/O_i
			OverlappingCommunityQuality
					.convertCrispToFuzzyOvCommunityWithNumComs(ovNodeCommunities);
		} else if (belongingVersion == 2) {
			// Crisp overlapping with belonging coefficients being the strength
			// of the node to the community
			OverlappingCommunityQuality
					.convertCrispToFuzzyOvCommunityWithNodeStrength(outNet,
							inNet, isUndirected, mapCommunities,
							ovNodeCommunities);
		}

		// Use type float to save memory
		float[] communitySizes = new float[numComs];
		HashMap<Integer, HashMap<Integer, Double>> communityWeights = new HashMap<Integer, HashMap<Integer, Double>>();
		// float[][] communityWeights = new float[numComs][numComs];
		HashMap<Integer, HashMap<Integer, Double>> communityDensities = new HashMap<Integer, HashMap<Integer, Double>>();
		// float[][] communityDensities = new float[numComs][numComs];

		// ///////////////////////////////////////////
		for (Map.Entry<Integer, HashMap<Integer, Double>> nodeItem : outNet
				.entrySet()) {
			int nodeId = nodeItem.getKey();
			HashMap<Integer, Double> nodeNbs = nodeItem.getValue();
			HashMap<Integer, Double> nodeComs = ovNodeCommunities.get(nodeId);

			// Traverse the neighboring nodes of this node
			for (Map.Entry<Integer, Double> nbItem : nodeNbs.entrySet()) {
				int nbId = nbItem.getKey();
				double weight = nbItem.getValue();
				HashMap<Integer, Double> nbComs = ovNodeCommunities.get(nbId);

				for (Map.Entry<Integer, Double> nodeComItem : nodeComs
						.entrySet()) {
					int nodeComId = nodeComItem.getKey();
					double nodeComFactor = nodeComItem.getValue();
					if (!communityWeights.containsKey(nodeComId)) {
						communityWeights.put(nodeComId,
								new HashMap<Integer, Double>());
					}
					if (!communityDensities.containsKey(nodeComId)) {
						communityDensities.put(nodeComId,
								new HashMap<Integer, Double>());
					}

					for (Map.Entry<Integer, Double> nbComItem : nbComs
							.entrySet()) {
						int nbComId = nbComItem.getKey();
						double nbComFactor = nbComItem.getValue();
						HashMap<Integer, Double> comWeight = communityWeights
								.get(nodeComId);
						if (!comWeight.containsKey(nbComId)) {
							comWeight.put(nbComId, 0.0);
						}
						HashMap<Integer, Double> comDensity = communityDensities
								.get(nodeComId);
						if (!comDensity.containsKey(nbComId)) {
							comDensity.put(nbComId, 0.0);
						}

						// Depends on belongingFunVersion
						if (belongingFunVersion == 0) {
							comDensity.put(nbComId, comDensity.get(nbComId)
									+ (nodeComFactor + nbComFactor) / 2);
							comWeight.put(nbComId, comWeight.get(nbComId)
									+ weight * (nodeComFactor + nbComFactor)
									/ 2);

							// communityDensities[nodeComId][nbComId] +=
							// (nodeComFactor + nbComFactor) / 2;
							// communityWeights[nodeComId][nbComId] += weight
							// * (nodeComFactor + nbComFactor) / 2;
						} else if (belongingFunVersion == 1) {
							comDensity.put(nbComId, comDensity.get(nbComId)
									+ nodeComFactor * nbComFactor);
							comWeight.put(nbComId, comWeight.get(nbComId)
									+ weight * nodeComFactor * nbComFactor);

							// communityDensities[nodeComId][nbComId] +=
							// nodeComFactor
							// * nbComFactor;
							// communityWeights[nodeComId][nbComId] += weight
							// * nodeComFactor * nbComFactor;
						} else if (belongingFunVersion == 2) {
							comDensity.put(nbComId, comDensity.get(nbComId)
									+ Math.max(nodeComFactor, nbComFactor));
							comWeight.put(
									nbComId,
									comWeight.get(nbComId)
											+ weight
											* Math.max(nodeComFactor,
													nbComFactor));

							// communityDensities[nodeComId][nbComId] +=
							// Math.max(
							// nodeComFactor, nbComFactor);
							// communityWeights[nodeComId][nbComId] += weight
							// * Math.max(nodeComFactor, nbComFactor);
						}
					}
				}
			} // node neighbors for
		} // nodes for
			// ///////////////////////////////////////

		for (int iComId = 0; iComId < numComs; ++iComId) {
			Set<Integer> iCommunity = mapCommunities.get(iComId);

			// Calculate community size with belonging factors
			for (int iNodeId : iCommunity) {
				communitySizes[iComId] += OverlappingCommunityQuality
						.getNodeBelongingFactor(ovNodeCommunities, iNodeId,
								iComId);
			}

			if (!communityWeights.containsKey(iComId)) {
				continue;
			}

			for (int jComId = 0; jComId < numComs; ++jComId) {

				// To save time, if there is no edge between the two
				// communities, we skip this pair
				// if (communityWeights[iComId][jComId] == 0) {
				// continue;
				// }

				if (!communityWeights.get(iComId).containsKey(jComId)) {
					continue;
				}

				Set<Integer> jCommunity = mapCommunities.get(jComId);

				// Get the numerator and denominator of community density
				// double numerator = 0;
				float denominator = 0;
				for (int iNodeId : iCommunity) {
					// HashMap<Integer, Double> iNbs = outNet.get(iNodeId);
					double iFactor = OverlappingCommunityQuality
							.getNodeBelongingFactor(ovNodeCommunities, iNodeId,
									iComId);
					for (int jNodeId : jCommunity) {
						double jFactor = OverlappingCommunityQuality
								.getNodeBelongingFactor(ovNodeCommunities,
										jNodeId, jComId);
						if (iComId == jComId) {
							if (jNodeId != iNodeId) {
								if (belongingFunVersion == 0) {
									denominator += (iFactor + jFactor) / 2;
								} else if (belongingFunVersion == 1) {
									denominator += iFactor * jFactor;
								} else if (belongingFunVersion == 2) {
									denominator += Math.max(iFactor, jFactor);
								}
							}
						} else {
							if (belongingFunVersion == 0) {
								denominator += (iFactor + jFactor) / 2;
							} else if (belongingFunVersion == 1) {
								denominator += iFactor * jFactor;
							} else if (belongingFunVersion == 2) {
								denominator += Math.max(iFactor, jFactor);
							}
						}
					} // jNodeId for
				} // iNodeId for

				HashMap<Integer, Double> comDensity = communityDensities
						.get(iComId);
				if (denominator == 0) {
					comDensity.put(jComId, 0.0);
					// communityDensities[iComId][jComId] = 0;
				} else {
					comDensity
							.put(jComId, comDensity.get(jComId) / denominator);
					// communityDensities[iComId][jComId] =
					// communityDensities[iComId][jComId]
					// / denominator;
				}
			} // j for
		} // i for

		// Compute average community quality of the sample
		double Q = 0;
		double NQ = 0;
		double Qds = 0;
		double intraEdges = 0;
		double intraDensity = 0;
		double contraction = 0;
		double interEdges = 0;
		// double interDensity = 0;
		double expansion = 0;
		double conductance = 0;
		// Fitness function
		double fitnessFunction = 0;

		// Average modularity degree metric
		double D = 0;

		for (int iComId = 0; iComId < numComs; ++iComId) {
			HashMap<Integer, Double> comWeight = communityWeights.get(iComId);
			HashMap<Integer, Double> comDensity = communityDensities
					.get(iComId);

			double iCommSize = communitySizes[iComId];
			double inWeights = 0;
			if (comWeight.containsKey(iComId)) {
				inWeights = comWeight.get(iComId);
			}
			// double inWeights = communityWeights[iComId][iComId];
			double inDensity = 0;
			if (comDensity.containsKey(iComId)) {
				inDensity = comDensity.get(iComId);
			}

			// double inDensity = communityDensities[iComId][iComId];
			// The number of outgoing edges from nodes inside the community to
			// the nodes ouside this community
			double outWeights = 0;
			// The number of incoming edges from nodes outside the community to
			// the nodes inside this community
			double out_incoming_weights = 0;
			double splitPenalty = 0;

			// The total weight of the subnetwork including the community and
			// its
			// neighboring communities
			double nebComTotalWeight = 0;
			HashSet<Integer> nebComs = new HashSet<Integer>();
			nebComs.add(iComId);

			for (int jComId = 0; jComId < numComs; ++jComId) {
				if (jComId != iComId) {
					double wij = 0;
					if (comWeight.containsKey(jComId)) {
						wij = comWeight.get(jComId);
					}

					// double wij = communityWeights[iComId][jComId];
					double wji = 0;
					if (communityWeights.get(jComId).containsKey(iComId)) {
						wji = communityWeights.get(jComId).get(iComId);
					}
					// double wji = communityWeights[jComId][iComId];
					outWeights += wij;
					if (wij != 0) {
						nebComs.add(jComId);
					}

					if (!isUndirected) {
						out_incoming_weights += wji;
						if (wji != 0) {
							nebComs.add(jComId);
						}
					}

					// double sp = (wij / totalWeight)
					// * communityDensities[iComId][jComId];
					double tmpDensity = 0;
					if (comDensity.containsKey(jComId)) {
						tmpDensity = comDensity.get(jComId);
					}
					double sp = wij * tmpDensity;
					// double sp = wij * communityDensities[iComId][jComId];
					splitPenalty += sp;
				}
			}

			// Calculate nebComTotalWeight
			for (int icom : nebComs) {
				for (int jcom : nebComs) {
					if (communityWeights.get(icom).containsKey(jcom)) {
						nebComTotalWeight += communityWeights.get(icom).get(
								jcom);
					}
					// nebComTotalWeight += communityWeights[icom][jcom];
				}
			}

			// System.out.println(inWeights + "\t" + outWeights + "\t"
			// + nebComTotalWeight + "\t" + nebNodeTotalWeight + "\t"
			// + wideNebNodeTotalWeight + "\t" + iCommSize + "\t"
			// + inDensity + "\t" + dc_out + "\t" + dc_out_denominator);

			if (isUndirected) {
				// modularity
				Q += inWeights / totalWeight
						- Math.pow((inWeights + outWeights) / totalWeight, 2);
				// Modularity Density Qds
				Qds += (inWeights / totalWeight)
						* inDensity
						- Math.pow(((inWeights + outWeights) / totalWeight)
								* inDensity, 2) - splitPenalty / totalWeight;

				if (nebComTotalWeight != 0) {
					NQ += inWeights
							/ nebComTotalWeight
							- Math.pow((inWeights + outWeights)
									/ nebComTotalWeight, 2);
				}
			} else {
				Q += inWeights
						/ totalWeight
						- ((inWeights + outWeights) * (inWeights + out_incoming_weights))
						/ Math.pow(totalWeight, 2);
				// Modularity Density Qds
				Qds += (inWeights / totalWeight)
						* inDensity
						- (((inWeights + outWeights) * (inWeights + out_incoming_weights)) / Math
								.pow(totalWeight, 2)) * Math.pow(inDensity, 2)
						- splitPenalty / totalWeight;

				if (nebComTotalWeight != 0) {
					NQ += inWeights
							/ nebComTotalWeight
							- ((inWeights + outWeights) * (inWeights + out_incoming_weights))
							/ Math.pow(nebComTotalWeight, 2);
				}
			}

			// intra-edges
			if (isUndirected) {
				intraEdges += inWeights / 2;
			} else {
				intraEdges += inWeights;
			}
			// contraction: average degree
			if (inWeights == 0 || iCommSize == 0) {
				contraction += 0;
			} else {
				contraction += inWeights / iCommSize;
			}
			// intra-density
			intraDensity += inDensity;
			interEdges += outWeights;
			// inter-density
			// if (numNodes == iCommSize) {
			// interDensity += 0;
			// } else {
			// interDensity += outWeights
			// / (iCommSize * (numNodes - iCommSize));
			// }
			if (outWeights == 0 || iCommSize == 0) {
				expansion += 0;
			} else {
				expansion += outWeights / iCommSize;
			}

			// Avoid that totalInterEdges==0 and communityEdges[i][i]==0
			if (outWeights == 0) {
				conductance += 0;
			} else {
				conductance += outWeights / (inWeights + outWeights);
			}

			// The fitness function
			if (inWeights == 0) {
				fitnessFunction += 0;
			} else {
				double fitness = 0;
				if (isUndirected) {
					fitness = inWeights / (inWeights + 2 * outWeights);
					fitnessFunction += fitness;
				} else {
					fitness = inWeights
							/ (inWeights + outWeights + out_incoming_weights);
					fitnessFunction += fitness;
				}
			}

			// Average modularity degree metric
			if (iCommSize == 0) {
				D += 0;
			} else {
				D += (inWeights - outWeights) / iCommSize;
			}
		} // for

		double[] qualities = { Q, NQ, Qds, intraEdges / numComs,
				intraDensity / numComs, contraction / numComs,
				interEdges / numComs, expansion / numComs,
				conductance / numComs, fitnessFunction / numComs, D };

		// long endTime = System.currentTimeMillis();
		// System.out.println("running time: " + (endTime - startTime) + "ms");

		return qualities;
	}

	/**
	 * Overlapping community quality metrics (QovL) without ground truth
	 * communities
	 * 
	 * @param networkFile
	 * @param isUnweighted
	 * @param isUndirected
	 * @param communityFile
	 * @param belongingVersion
	 *            0: fuzzy overlapping; 1: crisp overlapping with belonging
	 *            coefficients being 1/O_i; 2: crisp overlapping with belonging
	 *            coefficients being the strength of the node to the community.
	 * @param belongingFunVersion
	 *            0: average; 1: product; 2: max; 3: the paper default.
	 * @return
	 */
	public static double computeQovLink(String networkFile,
			boolean isUnweighted, boolean isUndirected, String communityFile,
			int belongingVersion, int belongingFunVersion) {
		// long startTime = System.currentTimeMillis();

		// Get outgoing network
		HashMap<Integer, HashMap<Integer, Double>> outNet = new HashMap<Integer, HashMap<Integer, Double>>();
		// Return total weight. if undirected: 2m; if directed: m
		double[] weights = CommunityQuality.getNetwork(networkFile,
				isUnweighted, isUndirected, outNet);
		double totalWeight = weights[0];
		// double maxWeight = weights[1];
		long numNodes = outNet.size();

		// System.out.println("#node = " + numNodes);

		// If network is directed, get incoming network
		HashMap<Integer, HashMap<Integer, Double>> inNet = null;
		if (!isUndirected) {
			inNet = new HashMap<Integer, HashMap<Integer, Double>>();
			CommunityQuality.getReversedNetwork(networkFile, isUnweighted,
					isUndirected, inNet);
		}

		// System.out.println("Finish reading the network.");
		Map<Integer, Set<Integer>> mapCommunities = CommunityQuality
				.getMapCommunities(communityFile);
		int numComs = mapCommunities.size();
		// System.out.println("#com = " + numComs);

		HashMap<Integer, HashMap<Integer, Double>> ovNodeCommunities = OverlappingCommunityQuality
				.getCrispOverlappingNodeCommunities(communityFile);

		if (belongingVersion == 1) {
			// Crisp overlapping with belonging coefficients being 1/O_i
			OverlappingCommunityQuality
					.convertCrispToFuzzyOvCommunityWithNumComs(ovNodeCommunities);
		} else if (belongingVersion == 2) {
			// Crisp overlapping with belonging coefficients being the strength
			// of the node to the community
			OverlappingCommunityQuality
					.convertCrispToFuzzyOvCommunityWithNodeStrength(outNet,
							inNet, isUndirected, mapCommunities,
							ovNodeCommunities);
		}

		double QovL = 0;
		for (int comId = 0; comId < numComs; ++comId) {
			Set<Integer> community = mapCommunities.get(comId);
			HashMap<Integer, Double> expectedBetas = new HashMap<Integer, Double>();

			// Compute expectedBetas
			for (int comNodeId : community) {
				double comNodeFactor = OverlappingCommunityQuality
						.getNodeBelongingFactor(ovNodeCommunities, comNodeId,
								comId);
				double expectedBeta = 0;
				// To save time here, we do not consider all the nodes in the
				// network but only consider the nodes in the community since
				// when belonging coefficient = 0, the F function (in the paper,
				// here bf = 3) is almost 0, so it's meaningless to consider the
				// nodes that have belonging coefficients equal to 0
				// for (int netNodeId : outNet.keySet()) {
				for (int netNodeId : community) {
					double netNodeFactor = OverlappingCommunityQuality
							.getNodeBelongingFactor(ovNodeCommunities,
									netNodeId, comId);
					if (belongingFunVersion == 0) {
						expectedBeta += (comNodeFactor + netNodeFactor)
								/ (2 * numNodes);
					} else if (belongingFunVersion == 1) {
						expectedBeta += (comNodeFactor * netNodeFactor)
								/ numNodes;
					} else if (belongingFunVersion == 2) {
						expectedBeta += Math.max(comNodeFactor, netNodeFactor)
								/ numNodes;
					} else if (belongingFunVersion == 3) {
						expectedBeta += OverlappingCommunityQuality
								.getQovLinkFunction(comNodeFactor,
										netNodeFactor)
								/ numNodes;
					}
				}
				expectedBetas.put(comNodeId, expectedBeta);
			}

			for (int iNodeId : community) {
				double iFactor = OverlappingCommunityQuality
						.getNodeBelongingFactor(ovNodeCommunities, iNodeId,
								comId);
				// Get ikout
				HashMap<Integer, Double> iNbs = outNet.get(iNodeId);
				double ikout = 0;
				for (Map.Entry<Integer, Double> iNbEntry : iNbs.entrySet()) {
					ikout += iNbEntry.getValue();
				}

				for (int jNodeId : community) {
					double jFactor = OverlappingCommunityQuality
							.getNodeBelongingFactor(ovNodeCommunities, jNodeId,
									comId);
					// Get jkout and jbetaout
					HashMap<Integer, Double> jNbs = outNet.get(jNodeId);
					if (!isUndirected) {
						jNbs = inNet.get(jNodeId);
					}
					double jkin = 0;
					for (Map.Entry<Integer, Double> jNbEntry : jNbs.entrySet()) {
						jkin += jNbEntry.getValue();
					}

					double Aij = 0;
					if ((iNodeId != jNodeId) && (iNbs.containsKey(jNodeId))) {
						Aij = iNbs.get(jNodeId);
					}

					double ijbeta = 0;
					if (belongingFunVersion == 0) {
						ijbeta = (iFactor + jFactor) / 2;
					} else if (belongingFunVersion == 1) {
						ijbeta = iFactor * jFactor;
					} else if (belongingFunVersion == 2) {
						ijbeta = Math.max(iFactor, jFactor);
					} else if (belongingFunVersion == 3) {
						ijbeta = OverlappingCommunityQuality
								.getQovLinkFunction(iFactor, jFactor);
					}

					// System.out.println(iNodeId + "\t" + jNodeId + "\t" +
					// ijbeta
					// + "\t" + expectedBetas.get(iNodeId) + "\t"
					// + expectedBetas.get(jNodeId));
					QovL += ijbeta
							* Aij
							- (expectedBetas.get(iNodeId) * ikout
									* expectedBetas.get(jNodeId) * jkin)
							/ totalWeight;
				} // jNode for
			} // iNode for
		} // comId for

		QovL /= totalWeight;
		// System.out.println("QovL = " + QovL);
		return QovL;
	}

	/**
	 * 
	 * @param iFactor
	 * @param jFactor
	 * @return
	 */
	public static double getQovLinkFunction(double iFactor, double jFactor) {
		double fvalue = 0;
		int p = 30;
		fvalue = 1 / ((1 + Math.exp(p - 2 * p * iFactor)) * (1 + Math.exp(p - 2
				* p * jFactor)));
		return fvalue;
	}

	/**
	 * Compute the NMI for two covers (overlapping communities)
	 * 
	 * @param disCommunityFile
	 *            : the overlapping communities detected
	 * @param realCommunityFile
	 *            : the ground truth communities
	 * @return
	 */
	public static double computeNMI(String disCommunityFile,
			String realCommunityFile) {
		List<Set<Integer>> discoveredCommunities = CommunityQuality
				.getCommunities(disCommunityFile);
		List<Set<Integer>> realCommunities = CommunityQuality
				.getCommunities(realCommunityFile);
		int exsize = discoveredCommunities.size();
		int realsize = realCommunities.size();

		// System.out.println("exSize = " + exsize + ", realSize = " +
		// realsize);

		// When there is no community detected or only 1 ground truth community,
		// nmi = 0.
		if (exsize <= 1 || realsize <= 1) {
			return 0;
		}

		// total number of nodes
		double numNodes = 0;
		HashSet<Integer> nodes = new HashSet<Integer>();
		for (int i = 0; i < realsize; ++i) {
			Set<Integer> realCommunity = realCommunities.get(i);
			nodes.addAll(realCommunity);
		}
		numNodes = nodes.size();
		nodes.clear();

		// Calculate H(X | Y)_norm
		double HX_with_Y_norm = 0;
		for (int i = 0; i < realsize; ++i) {
			Set<Integer> realCommunity = realCommunities.get(i);
			int realComSize = realCommunity.size();

			double minEntropy = Double.MAX_VALUE;
			for (int j = 0; j < exsize; ++j) {
				Set<Integer> exCommunity = discoveredCommunities.get(j);
				int exComSize = exCommunity.size();

				// Calculate H(X_k, Y_l)
				int intersect = CommunityQuality.getIntersectionNumber(
						realCommunity, exCommunity);
				int union = CommunityQuality.getUnionNumber(realCommunity,
						exCommunity);
				double HXk_Yl = logUtil(intersect / numNodes)
						+ logUtil((realComSize - intersect) / numNodes)
						+ logUtil((exComSize - intersect) / numNodes)
						+ logUtil((numNodes - union) / numNodes);

				// calculate H(Yl)
				double HYl = logUtil(exComSize / numNodes)
						+ logUtil(1 - exComSize / numNodes);

				// calculate H(X_k | Y_l)
				double HXk_with_Yl = HXk_Yl - HYl;
				double constraint = logUtil(intersect / numNodes)
						+ logUtil((numNodes - union) / numNodes)
						- logUtil((realComSize - intersect) / numNodes)
						- logUtil((exComSize - intersect) / numNodes);

				// System.out.println("intersect = " + intersect + ", union = "
				// + union + ", HXk_Yl = " + HXk_Yl + ", HYl = " + HYl
				// + ", HXk_with_Yl = " + HXk_with_Yl + ", constraint = "
				// + constraint);

				// Record the minimum entropy
				if ((HXk_with_Yl < minEntropy) && (constraint > 0)) {
					minEntropy = HXk_with_Yl;
				}
			}

			// calculate H(X_k)
			double HXk = logUtil(realComSize / numNodes)
					+ logUtil(1 - realComSize / numNodes);

			// H(X_k | Y)_norm: according to procedure 2 of Lancichinetti's
			// paper
			double HXk_with_Y_norm = 1;
			// When real or detected community results have only 1 community
			// each, then this value will be "NaN".
			// double HXk_with_Y_norm = HXk / HXk;

			// There is case that minEntropy==0 when there are two
			// communities that exactly match with each other
			if (minEntropy < Double.MAX_VALUE) {
				// if (HXk == 0) {
				// System.out.println("HXk=0, minEntropy = " + minEntropy);
				// }

				if (minEntropy == 0 && HXk == 0) {
					// Set to 1 to make sure that when there is only 1 community
					// in ground truth or in detected result, NMI will be almost
					// 0
					HXk_with_Y_norm = 1;
				} else {
					HXk_with_Y_norm = minEntropy / HXk;
				}
			}

			// System.out.println("Real: minEntropy = " + minEntropy +
			// ", HXk = "
			// + HXk + ", HXk_with_Y_norm = " + HXk_with_Y_norm + "\n");
			// Get total H(X | Y)_norm
			HX_with_Y_norm += HXk_with_Y_norm / realsize;
		} // for real coms

		// Calculate H(Y | X)_norm, the same procedure with calculating H(X |
		// Y)_norm
		double HY_with_X_norm = 0;
		for (int j = 0; j < exsize; ++j) {
			Set<Integer> exCommunity = discoveredCommunities.get(j);
			int exComSize = exCommunity.size();

			double minEntropy = Double.MAX_VALUE;
			for (int i = 0; i < realsize; ++i) {
				Set<Integer> realCommunity = realCommunities.get(i);
				int realComSize = realCommunity.size();

				// Calculate H(Y_l, X_k)
				int intersect = CommunityQuality.getIntersectionNumber(
						exCommunity, realCommunity);
				int union = CommunityQuality.getUnionNumber(exCommunity,
						realCommunity);
				double HYl_Xk = logUtil(intersect / numNodes)
						+ logUtil((exComSize - intersect) / numNodes)
						+ logUtil((realComSize - intersect) / numNodes)
						+ logUtil((numNodes - union) / numNodes);

				// calculate H(Xk)
				double HXk = logUtil(realComSize / numNodes)
						+ logUtil(1 - realComSize / numNodes);

				// calculate H(Y_l | X_k)
				double HYl_with_Xk = HYl_Xk - HXk;
				double constraint = logUtil(intersect / numNodes)
						+ logUtil((numNodes - union) / numNodes)
						- logUtil((exComSize - intersect) / numNodes)
						- logUtil((realComSize - intersect) / numNodes);

				// if ((HYl_with_Xk < minEntropy) && (constraint < 0)) {
				// System.out.println("HYl_with_Xk=" + HYl_with_Xk
				// + ", minEntropy=" + minEntropy + ", constraint="
				// + constraint);
				// }

				// Record the minimum entropy
				if ((HYl_with_Xk < minEntropy) && (constraint > 0)) {
					minEntropy = HYl_with_Xk;
				}
			}

			// calculate H(Y_l)
			double HYl = logUtil(exComSize / numNodes)
					+ logUtil(1 - exComSize / numNodes);

			// H(Y_l | X)_norm: according to procedure 2 of Lancichinetti's
			// paper
			double HYl_with_X_norm = 1;
			// When real or detected community results have only 1 community
			// each, then this value will be "NaN".
			// double HYl_with_X_norm = HYl / HYl;
			if (minEntropy < Double.MAX_VALUE) {
				if (minEntropy == 0 && HYl == 0) {
					// Set to 1 to make sure that when there is only 1 community
					// in ground truth or in detected result, NMI will be almost
					// 0
					HYl_with_X_norm = 1;
				} else {
					HYl_with_X_norm = minEntropy / HYl;
				}
			}

			// System.out.println("Dis " + HYl_with_X_norm);
			// Get total H(Y | X)_norm
			HY_with_X_norm += HYl_with_X_norm / exsize;
		}

		// Calculate nmi
		double nmi = 1 - (1 / 2.0) * (HX_with_Y_norm + HY_with_X_norm);
		return nmi;
	}

	/**
	 * Compute the NMI for two covers (overlapping communities)
	 * 
	 * @param disCommunityFile
	 *            : the overlapping communities detected
	 * @param realCommunityFile
	 *            : the ground truth communities
	 * @return
	 */
	public static double computeNMI(List<Set<Integer>> discoveredCommunities,
			List<Set<Integer>> realCommunities) {
		int exsize = discoveredCommunities.size();
		int realsize = realCommunities.size();

		// When there is no community detected or only 1 ground truth community,
		// nmi = 0.
		if (exsize <= 1 || realsize <= 1) {
			return 0;
		}

		// total number of nodes
		double numNodes = 0;
		HashSet<Integer> nodes = new HashSet<Integer>();
		for (int i = 0; i < realsize; ++i) {
			Set<Integer> realCommunity = realCommunities.get(i);
			nodes.addAll(realCommunity);
		}
		numNodes = nodes.size();
		nodes.clear();

		// Calculate H(X | Y)_norm
		double HX_with_Y_norm = 0;
		for (int i = 0; i < realsize; ++i) {
			Set<Integer> realCommunity = realCommunities.get(i);
			int realComSize = realCommunity.size();

			double minEntropy = Double.MAX_VALUE;
			for (int j = 0; j < exsize; ++j) {
				Set<Integer> exCommunity = discoveredCommunities.get(j);
				int exComSize = exCommunity.size();

				// Calculate H(X_k, Y_l)
				int intersect = CommunityQuality.getIntersectionNumber(
						realCommunity, exCommunity);
				int union = CommunityQuality.getUnionNumber(realCommunity,
						exCommunity);
				double HXk_Yl = logUtil(intersect / numNodes)
						+ logUtil((realComSize - intersect) / numNodes)
						+ logUtil((exComSize - intersect) / numNodes)
						+ logUtil((numNodes - union) / numNodes);

				// calculate H(Yl)
				double HYl = logUtil(exComSize / numNodes)
						+ logUtil(1 - exComSize / numNodes);

				// calculate H(X_k | Y_l)
				double HXk_with_Yl = HXk_Yl - HYl;
				double constraint = logUtil(intersect / numNodes)
						+ logUtil((numNodes - union) / numNodes)
						- logUtil((realComSize - intersect) / numNodes)
						- logUtil((exComSize - intersect) / numNodes);

				// Record the minimum entropy
				if ((HXk_with_Yl < minEntropy) && (constraint > 0)) {
					minEntropy = HXk_with_Yl;
				}
			}

			// calculate H(X_k)
			double HXk = logUtil(realComSize / numNodes)
					+ logUtil(1 - realComSize / numNodes);

			// H(X_k | Y)_norm: according to procedure 2 of Lancichinetti's
			// paper
			double HXk_with_Y_norm = HXk / HXk;
			if (minEntropy < Double.MAX_VALUE) {
				HXk_with_Y_norm = minEntropy / HXk;
			}

			// Get total H(X | Y)_norm
			HX_with_Y_norm += HXk_with_Y_norm / realsize;
		}

		// Calculate H(Y | X)_norm, the same procedure with calculating H(X |
		// Y)_norm
		double HY_with_X_norm = 0;
		for (int j = 0; j < exsize; ++j) {
			Set<Integer> exCommunity = discoveredCommunities.get(j);
			int exComSize = exCommunity.size();

			double minEntropy = Double.MAX_VALUE;
			for (int i = 0; i < realsize; ++i) {
				Set<Integer> realCommunity = realCommunities.get(i);
				int realComSize = realCommunity.size();

				// Calculate H(Y_l, X_k)
				int intersect = CommunityQuality.getIntersectionNumber(
						exCommunity, realCommunity);
				int union = CommunityQuality.getUnionNumber(exCommunity,
						realCommunity);
				double HYl_Xk = logUtil(intersect / numNodes)
						+ logUtil((exComSize - intersect) / numNodes)
						+ logUtil((realComSize - intersect) / numNodes)
						+ logUtil((numNodes - union) / numNodes);

				// calculate H(Xk)
				double HXk = logUtil(realComSize / numNodes)
						+ logUtil(1 - realComSize / numNodes);

				// calculate H(Y_l | X_k)
				double HYl_with_Xk = HYl_Xk - HXk;
				double constraint = logUtil(intersect / numNodes)
						+ logUtil((numNodes - union) / numNodes)
						- logUtil((exComSize - intersect) / numNodes)
						- logUtil((realComSize - intersect) / numNodes);

				// if ((HYl_with_Xk < minEntropy) && (constraint < 0)) {
				// System.out.println("HYl_with_Xk=" + HYl_with_Xk
				// + ", minEntropy=" + minEntropy + ", constraint="
				// + constraint);
				// }

				// Record the minimum entropy
				if ((HYl_with_Xk < minEntropy) && (constraint > 0)) {
					minEntropy = HYl_with_Xk;
				}
			}

			// calculate H(Y_l)
			double HYl = logUtil(exComSize / numNodes)
					+ logUtil(1 - exComSize / numNodes);

			// H(Y_l | X)_norm: according to procedure 2 of Lancichinetti's
			// paper
			double HYl_with_X_norm = HYl / HYl;
			if (minEntropy < Double.MAX_VALUE) {
				HYl_with_X_norm = minEntropy / HYl;
			}

			// Get total H(Y | X)_norm
			HY_with_X_norm += HYl_with_X_norm / exsize;
		}

		// Calculate nmi
		double nmi = 1 - (1 / 2.0) * (HX_with_Y_norm + HY_with_X_norm);
		return nmi;
	}

	/**
	 * Compute Omege Index for two covers (overlapping communities)
	 * 
	 * @param disCommunityFile
	 *            : the overlapping communities detected
	 * @param realCommunityFile
	 *            : the ground truth communities
	 * @return
	 */
	public static double computeOmegaIndex(String disCommunityFile,
			String realCommunityFile) {
		// Calculate the nodes and the number of times of these nodes that are
		// in the same communities with a certain node, for each node in the
		// network under ground truth community structure
		List<Set<Integer>> realCommunities = CommunityQuality
				.getCommunities(realCommunityFile);
		HashMap<Integer, HashMap<Integer, Integer>> realNodeNumComs = new HashMap<Integer, HashMap<Integer, Integer>>();
		for (Set<Integer> realCom : realCommunities) {
			// System.out.println(realCom.toString());

			for (int inode : realCom) {
				if (!realNodeNumComs.containsKey(inode)) {
					realNodeNumComs.put(inode, new HashMap<Integer, Integer>());
				}
				HashMap<Integer, Integer> nodeNumComs = realNodeNumComs
						.get(inode);
				for (int jnode : realCom) {
					if (inode == jnode) {
						continue;
					}

					if (nodeNumComs.containsKey(jnode)) {
						nodeNumComs.put(jnode, nodeNumComs.get(jnode) + 1);
					} else {
						nodeNumComs.put(jnode, 1);
					}
				}
			}
		}

		// Clear to save memory
		realCommunities.clear();

		// System.out.println(realNodeNumComs.toString());

		// Calculate the nodes and the number of times of these nodes that are
		// in the same communities with a certain node, for each node in the
		// network under detected community structure
		List<Set<Integer>> disCommunities = CommunityQuality
				.getCommunities(disCommunityFile);
		HashMap<Integer, HashMap<Integer, Integer>> disNodeNumComs = new HashMap<Integer, HashMap<Integer, Integer>>();
		for (Set<Integer> disCom : disCommunities) {
			// System.out.println(disCom.toString());

			for (int inode : disCom) {
				if (!disNodeNumComs.containsKey(inode)) {
					disNodeNumComs.put(inode, new HashMap<Integer, Integer>());
				}
				HashMap<Integer, Integer> nodeNumComs = disNodeNumComs
						.get(inode);
				for (int jnode : disCom) {
					if (inode == jnode) {
						continue;
					}

					if (nodeNumComs.containsKey(jnode)) {
						nodeNumComs.put(jnode, nodeNumComs.get(jnode) + 1);
					} else {
						nodeNumComs.put(jnode, 1);
					}
				}
			}
		}

		// System.out.println(disNodeNumComs.toString());

		long numNodes = realNodeNumComs.size();

		// System.out.println("numNodes = " + numNodes);

		// Record the number of pairs exist "realInter" times in both
		// real and detected covers
		HashMap<Integer, Double> commonNumComs = new HashMap<Integer, Double>();
		// Record the number of pairs exist "realInter" times in real
		// covers
		HashMap<Integer, Double> realNumComs = new HashMap<Integer, Double>();

		// Record the number of pairs exist "disInter" times in detected
		// covers
		HashMap<Integer, Double> disNumComs = new HashMap<Integer, Double>();

		double totalCommonNums = 0;
		double totalRealNums = 0;
		double totalDisNums = 0;

		for (Map.Entry<Integer, HashMap<Integer, Integer>> nodeItem : realNodeNumComs
				.entrySet()) {
			int inode = nodeItem.getKey();
			HashMap<Integer, Integer> realTmp = nodeItem.getValue();
			HashMap<Integer, Integer> disTmp = disNodeNumComs.get(inode);

			for (Map.Entry<Integer, Integer> numItem : realTmp.entrySet()) {
				int jnode = numItem.getKey();
				int numCommonComs = numItem.getValue();

				// System.out.println("real: " + inode + "\t" + jnode + "\t"
				// + numCommonComs);

				// Record the number of pairs exist "realInter" times in real
				// covers
				if (!realNumComs.containsKey(numCommonComs)) {
					realNumComs.put(numCommonComs, 1.0);
				} else {
					realNumComs.put(numCommonComs,
							realNumComs.get(numCommonComs) + 1);
				}
				++totalRealNums;

				if (disTmp.containsKey(jnode)) {
					int disNumCommonComs = disTmp.get(jnode);

					// System.out.println("dis: " + inode + "\t" + jnode + "\t"
					// + disNumCommonComs);
					// Record the number of pairs exist "realInter" times in
					// both
					// real and detected covers
					if (numCommonComs == disNumCommonComs) {
						// System.out.println("comm: " + inode + "\t" + jnode
						// + "\t" + disNumCommonComs);

						if (!commonNumComs.containsKey(numCommonComs)) {
							commonNumComs.put(numCommonComs, 1.0);
						} else {
							commonNumComs.put(numCommonComs,
									commonNumComs.get(numCommonComs) + 1);
						}
					}

					// Here, common means that the two have the same key
					++totalCommonNums;
				}
			}

			for (Map.Entry<Integer, Integer> numItem : disTmp.entrySet()) {
				int numCommonComs = numItem.getValue();

				// Record the number of pairs exist "disInter" times in detected
				// covers
				if (!disNumComs.containsKey(numCommonComs)) {
					disNumComs.put(numCommonComs, 1.0);
				} else {
					disNumComs.put(numCommonComs,
							disNumComs.get(numCommonComs) + 1);
				}
				++totalDisNums;
			}
		}

		double M = numNodes * (numNodes - 1.0);

		realNumComs.put(0, M - totalRealNums);
		disNumComs.put(0, M - totalDisNums);
		commonNumComs
				.put(0, M - totalRealNums - totalDisNums + totalCommonNums);

		// System.out.println("numNodes = " + numNodes + ", totalRealNums = "
		// + totalRealNums + ", totalDisNums = " + totalDisNums
		// + ", totalCommonNums = " + totalCommonNums);
		//
		// System.out.println(commonNumComs.toString());
		// System.out.println(realNumComs.toString());
		// System.out.println(disNumComs.toString());

		// Calculate Wu
		double wu = 0;
		for (Map.Entry<Integer, Double> commonItem : commonNumComs.entrySet()) {
			// System.out.println("times = " + commonItem.getKey()
			// + ", numComs = " + commonItem.getValue());
			wu += commonItem.getValue();
		}
		wu /= M;

		// System.out.println("");

		// Calculate we
		double we = 0;
		for (Map.Entry<Integer, Double> realItem : realNumComs.entrySet()) {
			// System.out.println("times = " + realItem.getKey() +
			// ", numComs = "
			// + realItem.getValue());

			int tmp = realItem.getKey();
			if (disNumComs.containsKey(tmp)) {
				we += realItem.getValue() * disNumComs.get(tmp);
			} else {
				we += 0;
			}
		}
		we /= Math.pow(M, 2);

		// Calculate Omega Index
		double omega = (wu - we) / (1 - we);
		return omega;
	}

	/**
	 * Fscore from paper: J. Xie, S. Kelley and B. K. Szymanski,
	 * "Overlapping Community Detection in Networks: the State of the Art and Comparative Study"
	 * , ACM Computing Surveys, 2013.
	 * 
	 * @param disCommunityFile
	 * @param realCommunityFile
	 * @return
	 */
	public static double[] computeFscore(String disCommunityFile,
			String realCommunityFile) {
		HashMap<Integer, HashSet<Integer>> realNodeCommunities = OverlappingCommunityQuality
				.getOverlappingNodeCommunities(realCommunityFile);
		HashMap<Integer, HashSet<Integer>> disNodeCommunities = OverlappingCommunityQuality
				.getOverlappingNodeCommunities(disCommunityFile);

		// Get overlapping nodes in ground truth communities
		HashSet<Integer> realNodes = new HashSet<Integer>();
		Map.Entry<Integer, HashSet<Integer>> item = null;
		Iterator<Map.Entry<Integer, HashSet<Integer>>> iter = realNodeCommunities
				.entrySet().iterator();
		while (iter.hasNext()) {
			item = iter.next();
			if (item.getValue().size() > 1) {
				realNodes.add(item.getKey());
			}
		}

		// Get overlapping nodes in detected communities
		HashSet<Integer> disNodes = new HashSet<Integer>();
		iter = disNodeCommunities.entrySet().iterator();
		while (iter.hasNext()) {
			item = iter.next();
			if (item.getValue().size() > 1) {
				disNodes.add(item.getKey());
			}
		}

		// Get fscore
		double numCorrected = CommunityQuality.getIntersectionNumber(realNodes,
				disNodes);

		// System.out.println("realNodes = " + realNodes + ", disNodes = "
		// + disNodes + ", numCorrected = " + numCorrected);

		double precision = 0;
		if (disNodes.size() > 0) {
			precision = numCorrected / disNodes.size();
		}

		double recall = 0;
		if (realNodes.size() > 0) {
			recall = numCorrected / realNodes.size();
		}

		double fscore = 0;
		if (precision > 0 && recall > 0) {
			fscore = (2 * precision * recall) / (precision + recall);
		}

		double[] fQualities = { precision, recall, fscore };
		return fQualities;
	}

	/**
	 * F1 measure from paper: Wu H, Gao L, Dong J, Yang X (2014) Detecting
	 * Overlapping Protein Complexes by Rough-Fuzzy Clustering in
	 * Protein-Protein Interaction Networks. PLoS One 9: e91856.
	 * 
	 * @param disCommunityFile
	 * @param realCommunityFile
	 * @return
	 */
	public static double[] computeF1(String disCommunityFile,
			String realCommunityFile) {
		Map<Integer, Set<Integer>> realCommunities = CommunityQuality
				.getMapCommunities(realCommunityFile);
		Map<Integer, Set<Integer>> disCommunities = CommunityQuality
				.getMapCommunities(disCommunityFile);
		double w = 0.25;
		int realNumComs = realCommunities.size();
		int disNumComs = disCommunities.size();

		double recall = 0;
		for (Set<Integer> realCommunity : realCommunities.values()) {
			int realSize = realCommunity.size();
			boolean flag = false;
			for (Set<Integer> disCommunity : disCommunities.values()) {
				int disSize = disCommunity.size();
				double intersect = CommunityQuality.getIntersectionNumber(
						realCommunity, disCommunity);
				double os = Math.pow(intersect, 2) / (realSize * disSize);
				if (os >= w) {
					flag = true;
					break;
				}
			}

			if (flag) {
				recall += 1;
			}
		}
		recall /= realNumComs;

		double precision = 0;
		for (Set<Integer> disCommunity : disCommunities.values()) {
			int disSize = disCommunity.size();
			boolean flag = false;
			for (Set<Integer> realCommunity : realCommunities.values()) {
				int realSize = realCommunity.size();
				double intersect = CommunityQuality.getIntersectionNumber(
						disCommunity, realCommunity);
				double os = Math.pow(intersect, 2) / (realSize * disSize);
				if (os >= w) {
					flag = true;
					break;
				}
			}

			if (flag) {
				precision += 1;
			}
		}
		precision /= disNumComs;

		double fscore = 0;
		if (precision > 0 && recall > 0) {
			fscore = (2 * precision * recall) / (precision + recall);
		}

		double[] fQualities = { precision, recall, fscore };
		return fQualities;
	}

	/**
	 * Compute entropy according to probability
	 * 
	 * @param p
	 * @return
	 */
	public static double logUtil(double p) {
		double value = 0;
		if (p > 0) {
			value = -p * (Math.log10(p) / Math.log10(2));
		}

		return value;
	}

	/**
	 * Return the belonging factor of a node to a community
	 * 
	 * @param ovNodeCommunities
	 * @param nodeId
	 * @param comId
	 * @return
	 */
	public static double getNodeBelongingFactor(
			HashMap<Integer, HashMap<Integer, Double>> ovNodeCommunities,
			int nodeId, int comId) {
		double factor = 0;
		HashMap<Integer, Double> communities = ovNodeCommunities.get(nodeId);
		if (communities.containsKey(comId)) {
			factor = communities.get(comId);
		}
		return factor;
	}

	/**
	 * 
	 * @param ovNodeCommunities
	 */
	public static void convertCrispToFuzzyOvCommunityWithNumComs(
			HashMap<Integer, HashMap<Integer, Double>> ovNodeCommunities) {
		for (Map.Entry<Integer, HashMap<Integer, Double>> nodeItem : ovNodeCommunities
				.entrySet()) {
			HashMap<Integer, Double> communities = nodeItem.getValue();
			double factor = 1.0 / communities.size();

			for (Map.Entry<Integer, Double> comItem : communities.entrySet()) {
				comItem.setValue(factor);
			}
		}
	}

	/**
	 * For undirected networks, only consider outgoing edges; for directed
	 * networks, consider both outgoing and incoming edges. Pay much attension
	 * to the overlapping situation.
	 * 
	 * @param outNet
	 * @param inNet
	 * @param isUndirected
	 * @param mapCommunities
	 * @param ovNodeCommunities
	 */
	public static void convertCrispToFuzzyOvCommunityWithNodeStrength(
			HashMap<Integer, HashMap<Integer, Double>> outNet,
			HashMap<Integer, HashMap<Integer, Double>> inNet,
			boolean isUndirected, Map<Integer, Set<Integer>> mapCommunities,
			HashMap<Integer, HashMap<Integer, Double>> ovNodeCommunities) {
		for (Map.Entry<Integer, HashMap<Integer, Double>> nodeItem : ovNodeCommunities
				.entrySet()) {
			int nodeId = nodeItem.getKey();
			HashMap<Integer, Double> communities = nodeItem.getValue();

			// Get the node strength
			HashMap<Integer, Double> outNodeNbs = outNet.get(nodeId);
			double totalNodeWeight = 0;
			for (Map.Entry<Integer, Double> nbItem : outNodeNbs.entrySet()) {
				int nbId = nbItem.getKey();
				double weight = nbItem.getValue();

				// Traverse each community this node belongs to to get the node
				// strength for this node to that community
				for (Map.Entry<Integer, Double> comItem : communities
						.entrySet()) {
					int comId = comItem.getKey();
					if (mapCommunities.get(comId).contains(nbId)) {
						comItem.setValue(comItem.getValue() + weight);
						totalNodeWeight += weight;
					}
				}
			}

			// if directed
			if (!isUndirected) {
				HashMap<Integer, Double> inNodeNbs = inNet.get(nodeId);
				for (Map.Entry<Integer, Double> nbItem : inNodeNbs.entrySet()) {
					int nbId = nbItem.getKey();
					double weight = nbItem.getValue();

					// Traverse each community this node belongs to to get the
					// node
					// strength for this node to that community
					for (Map.Entry<Integer, Double> comItem : communities
							.entrySet()) {
						int comId = comItem.getKey();
						if (mapCommunities.get(comId).contains(nbId)) {
							comItem.setValue(comItem.getValue() + weight);
							totalNodeWeight += weight;
						}
					}
				}
			}

			// divided by the total node weight
			for (Map.Entry<Integer, Double> comItem : communities.entrySet()) {
				// if (comItem.getValue() == 0 && totalNodeWeight == 0) {
				// System.out.println("NaN");
				// }

				if (comItem.getValue() == 0 || totalNodeWeight == 0) {
					comItem.setValue(0.0);
				} else {
					comItem.setValue(comItem.getValue() / totalNodeWeight);
				}
			}
		}
	}

	/**
	 * The id of communities start from 0
	 * 
	 * @param communityFile
	 * @return
	 */
	public static HashMap<Integer, HashMap<Integer, Double>> getCrispOverlappingNodeCommunities(
			String communityFile) {
		HashMap<Integer, HashMap<Integer, Double>> nodeCommunities = new HashMap<Integer, HashMap<Integer, Double>>();
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
					int node = Integer.parseInt(nodes[i]);

					if (!nodeCommunities.containsKey(node)) {
						nodeCommunities.put(node,
								new HashMap<Integer, Double>());
					}

					HashMap<Integer, Double> communities = nodeCommunities
							.get(node);
					communities.put(count, 0.0);
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

	/**
	 * The id of communities start from 0
	 * 
	 * @param communityFile
	 * @return
	 */
	public static HashMap<Integer, HashSet<Integer>> getOverlappingNodeCommunities(
			String communityFile) {
		HashMap<Integer, HashSet<Integer>> nodeCommunities = new HashMap<Integer, HashSet<Integer>>();
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
					int node = Integer.parseInt(nodes[i]);

					if (!nodeCommunities.containsKey(node)) {
						nodeCommunities.put(node, new HashSet<Integer>());
					}

					HashSet<Integer> communities = nodeCommunities.get(node);
					communities.add(count);
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

}
