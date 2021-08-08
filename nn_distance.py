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
    def __init__(self, points_ref, device, levels):
        """Builds the KDTree. See :ref:`build_kd_tree` for more details.
        """
        assert(device in ['gpu', 'cpu'])

        
        self.dtype = points_ref.dtype
        self.dims = points_ref.shape[-1]
        self.nr_ref_points = points_ref.shape[0]
        kdtree_str = "KDTree" + device.upper() + "%dD" % (self.dims) + ("F" if self.dtype == np.float32 else "")

        try:
            self.kdtree = getattr(cp_knn, kdtree_str)(points_ref, levels)
        except AttributeError as err:
            raise RuntimeError("Could not find the KD-Tree for your specified options. This probably means the library was not compiled for the specified dimensionality, precision or the targeted device.")

        self.structured_points = self.kdtree.get_structured_points()
        self.shuffled_ind = self.kdtree.get_shuffled_inds()
        self.use_gpu = use_gpu = (device == 'gpu')
        self.lib = (cp if use_gpu else np)
        self.dtype_idx = self.lib.int32

        if self.use_gpu:
            self.structured_points = cp.asarray(self.structured_points)
            self.shuffled_ind = cp.asarray(self.shuffled_ind)
        
    def _search_kd_tree_gpu(self, points_query, nr_nns_searches, result_dists, result_idx):
        cp_knn.searchKDTreeGPU(points_query, nr_nns_searches, self.part_nr, result_dists, result_idx)

    def _search_kd_tree_cpu(self, points_query, nr_nns_searches, result_dists, result_idx):
        cp_knn.searchKDTreeCPU(points_query, nr_nns_searches, self.part_nr, result_dists, result_idx)

    def query(self, points_query, nr_nns_searches=1, result_dists=None, result_idx=None):
        """Searches the specified KD-Tree for KNN of the given points

        Parameters
        ----------
        points_query : ndarray of float or double precision
            Points for which the KNNs will be computed
        nr_nns_searches : int, optional
            How many closest nearest neighbors will be queried (=k), by default 1
        result_dists : ndarray of float or double precision, optional
            Target array that will hold the resulting distance. If not specified, this will be dynamically created.
        result_idx : ndarray of dtype_idx type, optional
            Target array that will hold the resulting KNN indices. If not specified, this will be dynamically created.

        Returns
        -------
        tuple
            Returns the tuple containing

            * dists (ndarray of float or double precision) : Quadratic distance of KD-Tree points to the queried points
            * inds (ndarray of type int) : Indices of the K closest neighbors

        Raises
        ------
        RuntimeError
            If the requested KDTree can not be constructed.
        """
        if nr_nns_searches > self.nr_ref_points:
            raise RuntimeError("You requested more nearest neighbors than there are in the KD-Tree")

        if self.use_gpu and type(points_query) == np.ndarray:
            points_query = cp.asarray(points_query)

        if result_dists is None:
            result_dists = self.lib.empty(shape=[points_query.shape[0], nr_nns_searches], dtype=self.dtype)
        if result_idx is None:
            result_idx = self.lib.empty(shape=[points_query.shape[0], nr_nns_searches], dtype=self.dtype_idx)
            
        assert(list(result_dists.shape) == [points_query.shape[0], nr_nns_searches])
        assert(result_dists.dtype == self.dtype)
        assert(list(result_idx.shape) == [points_query.shape[0], nr_nns_searches])
        assert(result_idx.dtype == self.dtype_idx)

        for arr in [result_dists, result_idx, points_query]:
            if not arr.flags['C_CONTIGUOUS']:
                arr[:] = self.lib.ascontiguousarray(arr)

        get_ptr = lambda arr: (arr.data.ptr if self.use_gpu else arr.__array_interface__['data'][0])
        #Get pointer as int
        points_query_ptr = get_ptr(points_query)
        dists_ptr = get_ptr(result_dists)
        knn_idx_ptr = get_ptr(result_idx)

        self.kdtree.query(points_query_ptr, points_query.shape[0], nr_nns_searches, dists_ptr, knn_idx_ptr)
        return result_dists, self.shuffled_ind[result_idx]

def build_kd_tree(points_ref, device=None, levels=None):
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
    CPKDTree
        Returns a kdtree with a query method to find the nearest neighbors inside a point-cloud
    """
    if device is None:
        device = ('gpu' if gpu_available else 'cpu')

    if levels is None:
      levels = np.maximum(1, np.minimum(13, int(np.log(int(points_ref.shape[0])) / np.log(2))-3))
    
    assert(type(points_ref) == np.ndarray)
    assert(levels >= 1 and levels <= 13)
    assert(device != 'gpu' or gpu_available)
    assert(device in ['gpu', 'cpu'])

    return CPKDTree(points_ref, device, levels)
