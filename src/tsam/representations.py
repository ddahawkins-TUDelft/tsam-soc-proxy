# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from tsam.utils.durationRepresentation import durationRepresentation


def representations(
    candidates,
    clusterOrder,
    default,
    representationMethod=None,
    representationDict=None,
    distributionPeriodWise=True,
    timeStepsPerPeriod=None,
):
    clusterCenterIndices = None
    if representationMethod is None:
        representationMethod = default
    if representationMethod == "meanRepresentation":
        clusterCenters = meanRepresentation(candidates, clusterOrder)
    elif representationMethod == "medoidRepresentation":
        clusterCenters, clusterCenterIndices = medoidRepresentation(
            candidates, clusterOrder
        )
    elif representationMethod == "maxoidRepresentation":
        clusterCenters, clusterCenterIndices = maxoidRepresentation(
            candidates, clusterOrder
        )
    elif representationMethod == "minmaxmeanRepresentation":
        clusterCenters = minmaxmeanRepresentation(
            candidates, clusterOrder, representationDict, timeStepsPerPeriod
        )
    elif representationMethod == "durationRepresentation" or representationMethod == "distributionRepresentation":
        clusterCenters = durationRepresentation(
            candidates, clusterOrder, distributionPeriodWise, timeStepsPerPeriod, representMinMax=False,
        )
    elif representationMethod == "distributionAndMinMaxRepresentation":
        clusterCenters = durationRepresentation(
            candidates, clusterOrder, distributionPeriodWise, timeStepsPerPeriod, representMinMax=True,
        )
    elif representationMethod == "socRepresentation": #Dano
        clusterCenters, clusterCenterIndices = socRepresentation_with_diagnostics(
            candidates, clusterOrder, representationDict, timeStepsPerPeriod, representationAttribute='soc_stresses', plotDiagnostics=False
        )
    else:
        raise ValueError("Chosen 'representationMethod' does not exist.")
         
    return clusterCenters, clusterCenterIndices


def maxoidRepresentation(candidates, clusterOrder):
    """
    Represents the candidates of a given cluster group (clusterOrder)
    by its medoid, measured with the euclidean distance.

    :param candidates: Dissimilarity matrix where each row represents a candidate. required
    :type candidates: np.ndarray

    :param clusterOrder: Integer array where the index refers to the candidate and the
        Integer entry to the group. required
    :type clusterOrder: np.array
    """
    # set cluster member that is farthest away from the points of the other clusters as maxoid
    clusterCenters = []
    clusterCenterIndices = []
    for clusterNum in np.unique(clusterOrder):
        indice = np.where(clusterOrder == clusterNum)
        innerDistMatrix = euclidean_distances(candidates, candidates[indice])
        mindistIdx = np.argmax(innerDistMatrix.sum(axis=0))
        clusterCenters.append(candidates[indice][mindistIdx])
        clusterCenterIndices.append(indice[0][mindistIdx])

    return clusterCenters, clusterCenterIndices


def medoidRepresentation(candidates, clusterOrder):
    """
    Represents the candidates of a given cluster group (clusterOrder)
    by its medoid, measured with the euclidean distance.

    :param candidates: Dissimilarity matrix where each row represents a candidate. required
    :type candidates: np.ndarray

    :param clusterOrder: Integer array where the index refers to the candidate and the
        Integer entry to the group. required
    :type clusterOrder: np.array
    """
    # set cluster center as medoid
    clusterCenters = []
    clusterCenterIndices = []
    for clusterNum in np.unique(clusterOrder):
        indice = np.where(clusterOrder == clusterNum)
        innerDistMatrix = euclidean_distances(candidates[indice])
        mindistIdx = np.argmin(innerDistMatrix.sum(axis=0))
        clusterCenters.append(candidates[indice][mindistIdx])
        clusterCenterIndices.append(indice[0][mindistIdx])

    return clusterCenters, clusterCenterIndices


def meanRepresentation(candidates, clusterOrder):
    """
    Represents the candidates of a given cluster group (clusterOrder)
    by its mean.

    :param candidates: Dissimilarity matrix where each row represents a candidate. required
    :type candidates: np.ndarray

    :param clusterOrder: Integer array where the index refers to the candidate and the
        Integer entry to the group. required
    :type clusterOrder: np.array
    """
    # set cluster centers as means of the group candidates
    clusterCenters = []
    for clusterNum in np.unique(clusterOrder):
        indice = np.where(clusterOrder == clusterNum)
        currentMean = candidates[indice].mean(axis=0)
        clusterCenters.append(currentMean)
    return clusterCenters


def minmaxmeanRepresentation(
    candidates, clusterOrder, representationDict, timeStepsPerPeriod
):
    """
    Represents the candidates of a given cluster group (clusterOrder)
    by either the minimum, the maximum or the mean values of each time step for
    all periods in that cluster depending on the command for each attribute.

    :param candidates: Dissimilarity matrix where each row represents a candidate. required
    :type candidates: np.ndarray

    :param clusterOrder: Integer array where the index refers to the candidate and the
        Integer entry to the group. required
    :type clusterOrder: np.array

    :param representationDict: A dictionary which defines for each attribute whether the typical
        period should be represented by the minimum or maximum values within each cluster.
        optional (default: None)
    :type representationDict: dictionary

    :param timeStepsPerPeriod: The number of discrete timesteps which describe one period. required
    :type timeStepsPerPeriod: integer
    """
    # set cluster center depending of the representationDict
    clusterCenters = []
    for clusterNum in np.unique(clusterOrder):
        indice = np.where(clusterOrder == clusterNum)
        currentClusterCenter = np.zeros(len(representationDict) * timeStepsPerPeriod)
        for attributeNum in range(len(representationDict)):
            startIdx = attributeNum * timeStepsPerPeriod
            endIdx = (attributeNum + 1) * timeStepsPerPeriod
            if list(representationDict.values())[attributeNum] == "min":
                currentClusterCenter[startIdx:endIdx] = candidates[
                    indice, startIdx:endIdx
                ].min(axis=1)
            elif list(representationDict.values())[attributeNum] == "max":
                currentClusterCenter[startIdx:endIdx] = candidates[
                    indice, startIdx:endIdx
                ].max(axis=1)
            elif list(representationDict.values())[attributeNum] == "mean":
                currentClusterCenter[startIdx:endIdx] = candidates[
                    indice, startIdx:endIdx
                ].mean(axis=1)
            else:
                raise ValueError(
                    'At least one value in the representationDict is neither "min", "max" nor "mean".'
                )
        clusterCenters.append(currentClusterCenter)
    return clusterCenters

def socRepresentation(candidates, clusterOrder, representationDict, timeStepsPerPeriod): #Dano
    """
    Selects cluster representatives based on the timestep where `soc_stresses` is maximised.
    All features (demand, wind, etc.) are taken from the same candidate index.
    
    Parameters:
    - candidates: np.ndarray of shape (n_periods, n_features * timeStepsPerPeriod)
    - clusterOrder: 1D np.ndarray of length n_periods
    - representationDict: dict with feature names as keys (used to find soc_stresses)
    - timeStepsPerPeriod: int

    Returns:
    - clusterCenters: list of representative vectors (np.ndarray)
    - clusterCenterIndices: list of row indices
    """
    
    feature_names = list(representationDict.keys())
    try:
        soc_index = feature_names.index('soc_stresses')
    except ValueError:
        raise ValueError("'soc_stresses' must be present in representationDict.")

    clusterCenters = []
    clusterCenterIndices = []

    for clusterNum in np.unique(clusterOrder):
        indices = np.where(clusterOrder == clusterNum)[0]

        # Track maximum soc_stresses value and corresponding index
        max_val = -np.inf
        max_idx = None

        for i in indices:
            start = soc_index * timeStepsPerPeriod
            end = (soc_index + 1) * timeStepsPerPeriod
            current_max = np.max(candidates[i, start:end])

            if current_max > max_val:
                max_val = current_max
                max_idx = i

        # Add full vector for that timestep
        clusterCenters.append(candidates[max_idx])
        clusterCenterIndices.append(max_idx)

    return clusterCenters, clusterCenterIndices



def socRepresentation_with_diagnostics(
    candidates, clusterOrder, representationDict, timeStepsPerPeriod,
    representationAttribute='soc_stresses',
    plotDiagnostics=True
):
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('TkAgg') #avoids the annoying Qt errors on windows
    feature_names = list(representationDict.keys())
    try:
        soc_index = feature_names.index(representationAttribute)
    except ValueError:
        raise ValueError(f"'{representationAttribute}' must be present in representationDict.")

    clusterCenters = []
    clusterCenterIndices = []

    for clusterNum in np.unique(clusterOrder):
        indices = np.where(clusterOrder == clusterNum)[0]

        # For diagnostics
        score_val = -np.inf
        score_idx = None

        if plotDiagnostics:
            plt.figure(figsize=(10, 4))
            plt.title(f"Cluster {clusterNum} – SoC profiles")
        
        for idx in indices:
            start = soc_index * timeStepsPerPeriod
            end = (soc_index + 1) * timeStepsPerPeriod
            soc_profile = candidates[idx, start:end]
            if representationDict['soc_stresses'] == 'mean':
                current_score = np.mean(soc_profile)
            elif representationDict['soc_stresses'] == 'max':
                current_score = np.max(soc_profile)
            else: 
                current_score = np.min(soc_profile)

            if current_score > score_val:
                score_val = current_score
                score_idx = idx

            if plotDiagnostics:
                plt.plot(soc_profile, alpha=0.4, label=f"Idx {idx}")

        # Plot chosen one in bold
        if plotDiagnostics and score_idx is not None:
            chosen_profile = candidates[score_idx, soc_index * timeStepsPerPeriod : (soc_index + 1) * timeStepsPerPeriod]
            plt.plot(chosen_profile, label=f"Chosen: {score_idx}", color='black', linewidth=2)
            # plt.legend()
            plt.xlabel("Timestep")
            plt.ylabel("SoC stresses")
            plt.tight_layout()
            plt.show() 

        clusterCenters.append(candidates[score_idx])
        clusterCenterIndices.append(score_idx)

    return clusterCenters, clusterCenterIndices
