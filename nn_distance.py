import os
import numpy as np

try:
    import cupy as cp
    gpu_available = True
except ImportError as err:
    print("Import of cupy failed. Only the CPU version will be available.\nError Message: %s" % (err))
    gpu_available = False

import cp_knn

if gpu_available:
    gpu_available = cp_knn.check_for_gpu()
    if not gpu_available:
        print("The library was not successfully compiled using CUDA. Only the CPU version will be available.")

class CPKDTree:
    def __init__(self, structured_points, part_nr, shuffled_ind, device):
        assert(device in ['gpu', 'cpu'])

        self.dtype = structured_points.dtype
        self.dtype_idx = np.int32
        self.structured_points = structured_points
        self.part_nr = part_nr
        self.shuffled_ind = shuffled_ind
        self.use_gpu = use_gpu = (device == 'gpu')
        self.kdtree_func = (self._search_kd_tree_gpu if use_gpu else self._search_kd_tree_cpu)
        self.lib = (cp if use_gpu else np)
        
    def _search_kd_tree_gpu(self, points_query, nr_nns_searches, result_dists, result_idx):
        cp_knn.searchKDTreeGPU(points_query, nr_nns_searches, self.part_nr, result_dists, result_idx)

    def _search_kd_tree_cpu(self, points_query, nr_nns_searches, result_dists, result_idx):
        cp_knn.searchKDTreeCPU(points_query, nr_nns_searches, self.part_nr, result_dists, result_idx)

    def search_kd_tree(self, points_query, nr_nns_searches=1, result_dists=None, result_idx=None):
        if result_dists is None:
            result_dists = self.lib.empty(shape=[points_query.shape[0], nr_nns_searches], dtype=self.dtype)
        if result_idx is None:
            result_idx = self.lib.empty(shape=[points_query.shape[0], nr_nns_searches], dtype=self.dtype_idx)
            
        assert(result_dists.shape == [points_query.shape[0], nr_nns_searches])
        assert(result_dists.dtype == self.dtype)
        assert(result_idx.shape == [points_query.shape[0], nr_nns_searches])
        assert(result_idx.dtype == self.dtype_idx)

        if not result_dists.flags['C_CONTIGUOUS']:
            result_dists = self.lib.ascontiguousarray(result_dists)

        if not result_idx.flags['C_CONTIGUOUS']:
            result_idx = self.lib.ascontiguousarray(result_idx)

        #Get pointer as int
        #pointer, read_only_flag = a.__array_interface__['data'] #Numpy
        #a.data.ptr #Cupy

        self.kdtree_func(points_query, self.part_nr, nr_nns_searches, result_dists, result_idx)
        return result_dists, self.shuffled_ind[result_idx]

def build_kd_tree(points_ref, device=None, levels=None, **kwargs):
    """Builds the KD-Tree for subsequent queries using searchKDTree

    Builds the KD-Tree for subsequent queries using searchKDTree. Note that the 
    tree is always built on the CPU and then transferred to the GPU if necessary.

    Parameters
    ----------
    points_ref : numpy ndarray of float or double precision
        Points from which to build the KD-Tree
    device : string
        Specify either gpu or cpu here, depending on where you want to have the KD-Tree. If None is specified,
        will automatically pick the gpu, or the cpu if no gpu is available.
    levels : int, optional
        Levels of the KD-Tree (currently between 1 and 13 levels). If None is specified, will pick an appropriate value.

    Returns
    -------
    tuple
        Returns a triplet 
        
        * structured_points: points_ref, ordered by the KD-Tree structure
        * part_nr: Unique ID of the KD-Tree to later refer to the created tree
        * shuffled_ind: Indices to map structured_points -> points_ref
    """
    if device is None:
        device = ('gpu' if gpu_available else 'cpu')
    
    assert(type(points_ref) == np.ndarray)
    assert(levels >= 1 and levels <= 13)
    assert(device != 'gpu' or gpu_available)
    assert(device in ['gpu', 'cpu'])
    if levels is None:
      levels = np.maximum(1, np.minimum(13, int(np.log(int(points_ref.shape[0])) / np.log(2))-3))

    structured_points, part_nr, shuffled_ind = (cp_knn.buildKDTreeGPU(points_ref, levels=levels, **kwargs) if device == 'gpu' else cp_knn.buildKDTreeCPU(points_ref, levels=levels, **kwargs))
    return CPKDTree(structured_points, part_nr, shuffled_ind, device)


def search_kd_tree(points_query, metadata_address_kdtree, nr_nns_searches=1, shuffled_inds=None, **kwargs):
    """Searches the specified KD-Tree for KNN of the given points

    Parameters
    ----------
    points_query : tensor or array of float or double precision
        Points for which the KNNs will be computed
    metadata_address_kdtree : int
        Unique ID of the KD-Tree to be queried (see buildKDTree)
    nr_nns_searches : int, optional
        How many closest nearest neighbors will be queried (=k), by default 1
    shuffled_inds : tensor or array of type int, optional
        When creating the tree using buildKDTree, this array is returned to map
        the indices from structured_points, back to the original indices.
        If none, this remapping will not be performed and the returned indices
        map to the indices in structured_points.

    Returns
    -------
    tuple
        Returns the tuple containing

        * dists (tensor of float or double precision) : Quadratic distance of KD-Tree points to the queried points
        * inds (tensor of type int) : Indices of the K closest neighbors
    """
    dists, inds = _op_library.kd_tree_knn_search(points_query, metadata_address_kdtree=metadata_address_kdtree, 
                nr_nns_searches=nr_nns_searches, **kwargs)

    if shuffled_inds is not None:
        inds = tf.gather(shuffled_inds, tf.cast(inds, tf.int32))

    return dists, inds
