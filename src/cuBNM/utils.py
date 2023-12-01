import numpy as np
import scipy
import GPUtil

def avail_gpus():
    """
    Gets the number of available GPUs
    """
    return len(GPUtil.getAvailable())

def fc_norm_euclidean(x, y):
    """
    Calculates Euclidean distance of two FC arrays
    divided by their maximum possible distance, equal
    to the distance of np.ones(n_pairs) and -np.ones(n_pairs)
    or 2 * np.sqrt(n_pairs)
    """
    euclidean = scipy.spatial.distance.euclidean(x, y)
    max_euclidean = 2 * np.sqrt(x.size)
    return euclidean / max_euclidean
