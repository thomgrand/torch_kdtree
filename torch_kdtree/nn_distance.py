#import os
from typing import Union
import torch
import numpy as np
from . import torch_knn

gpu_available = torch_knn.check_for_gpu()
if not gpu_available:
    print("The library was not successfully compiled using CUDA. Only the CPU version will be available.")

_transl_torch_device = {"cpu": "CPU", "cuda": "GPU"}

class TorchKDTree:
    def __init__(self, points_ref : torch.Tensor, device : torch.device, levels : int, squared_distances : bool):
        """Builds the KDTree. See :ref:`build_kd_tree` for more details.
        """
        assert(device.type in ['cpu', 'cuda'])
        assert points_ref.shape[0] < 2**31, "Only 32 bit signed indexing implemented"
        
        self.dtype = points_ref.dtype
        self.dims = points_ref.shape[-1]
        self.nr_ref_points = points_ref.shape[0]
        kdtree_str = "KDTree" + _transl_torch_device[device.type] + "%dD" % (self.dims) + ("F" if self.dtype == torch.float32 else "")

        try:
            self.kdtree = getattr(torch_knn, kdtree_str)(points_ref.detach().cpu().numpy(), levels)
        except AttributeError as err:
            raise RuntimeError("Could not find the KD-Tree for your specified options. This probably means the library was not compiled for the specified dimensionality, precision or the targeted device. Original error:", err)

        self.structured_points = torch.from_numpy(self.kdtree.get_structured_points())
        self.shuffled_ind = torch.from_numpy(self.kdtree.get_shuffled_inds()).long()
        self.use_gpu = use_gpu = (device.type == 'cuda')
        self.device = device
        self.dtype_idx = torch.int32 #Restriction in the compiled library
        self.ref_requires_grad = points_ref.requires_grad
        self.points_ref_bak = points_ref #.clone()
        self.squared_distances = squared_distances

        if self.use_gpu:
            self.structured_points = self.structured_points.to(self.device)
            self.shuffled_ind = self.shuffled_ind.to(self.device)
        
    def _search_kd_tree_gpu(self, points_query, nr_nns_searches, result_dists, result_idx):
        torch_knn.searchKDTreeGPU(points_query, nr_nns_searches, self.part_nr, result_dists, result_idx)

    def _search_kd_tree_cpu(self, points_query, nr_nns_searches, result_dists, result_idx):
        torch_knn.searchKDTreeCPU(points_query, nr_nns_searches, self.part_nr, result_dists, result_idx)

    def query(self, points_query : torch.Tensor, nr_nns_searches : int=1, 
                result_dists : torch.Tensor=None, result_idx : torch.Tensor=None):
        """Searches the specified KD-Tree for KNN of the given points

        Parameters
        ----------
        points_query : torch.Tensor of float or double precision
            Points for which the KNNs will be computed
        nr_nns_searches : int, optional
            How many closest nearest neighbors will be queried (=k), by default 1
        result_dists : torch.Tensor of float or double precision, optional
            Target array that will hold the resulting distance. If not specified, this will be dynamically created.
        result_idx : torch.Tensor of dtype_idx type, optional
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

        points_query = points_query.to(self.device)

        if result_dists is None:
            result_dists = torch.empty(size=[points_query.shape[0], nr_nns_searches], dtype=self.dtype, device=self.device)
        if result_idx is None:
            result_idx = torch.empty(size=[points_query.shape[0], nr_nns_searches], dtype=self.dtype_idx, device=self.device)
            
        assert(list(result_dists.shape) == [points_query.shape[0], nr_nns_searches])
        assert(result_dists.dtype == self.dtype)
        assert(list(result_idx.shape) == [points_query.shape[0], nr_nns_searches])
        assert(result_idx.dtype == self.dtype_idx)
        assert(points_query.dtype == self.dtype)

        if not result_dists.is_contiguous():
            result_dists = result_dists.contiguous()

        if not result_idx.is_contiguous():
            result_idx = result_idx.contiguous()

        if not points_query.is_contiguous():
            points_query = points_query.contiguous()

        #Get pointer as int
        points_query_ptr = points_query.data_ptr()
        dists_ptr = result_dists.data_ptr()
        knn_idx_ptr = result_idx.data_ptr()

        self.kdtree.query(points_query_ptr, points_query.shape[0], nr_nns_searches, dists_ptr, knn_idx_ptr)
        dists = result_dists
        inds = self.shuffled_ind[result_idx.long()]

        if (points_query.requires_grad or self.ref_requires_grad) and torch.is_grad_enabled():
            dists = torch.sum((points_query[:, None] - self.points_ref_bak[inds])**2, dim=-1)

        if not self.squared_distances:
            dists = torch.sqrt(dists)

        return dists, inds

def build_kd_tree(points_ref : Union[torch.Tensor, np.ndarray], device : torch.device = None,
                    squared_distances = True, levels : int=None):
    """Builds the KD-Tree for subsequent queries using searchKDTree

    Builds the KD-Tree for subsequent queries using searchKDTree. Note that the 
    tree is always built on the CPU and then transferred to the GPU if necessary.

    Parameters
    ----------
    points_ref : torch.Tensor
        Points from which to build the KD-Tree
    device : torch.device
        Specify a target torch device where the KD-Tree will be located.
        Will automatically pick points_ref.device if not specified.
    squared_distances : bool
        If true, the squared euclidean distances will be returned, by default True,
    levels : int, optional
        Levels of the KD-Tree (currently between 1 and 13 levels). If None is specified, will pick an appropriate value.

    Returns
    -------
    TorchKDTree
        Returns a kdtree with a query method to find the nearest neighbors inside a point-cloud
    """
    if device is None:
        device = points_ref.device

    if levels is None:
      levels = np.maximum(1, np.minimum(13, int(np.log(int(points_ref.shape[0])) / np.log(2))-3))

    if issubclass(type(points_ref), np.ndarray):
        points_ref = torch.from_numpy(points_ref)

    if issubclass(type(device), str):
        device = torch.device(device)
    
    assert(levels >= 1 and levels <= 13)
    assert issubclass(type(points_ref), torch.Tensor)
    assert device.type != 'cuda' or gpu_available, "You requested the KD-Tree on the GPU, but the library was compiled with CPU support only"
    assert(device.type in ['cuda', 'cpu'])

    return TorchKDTree(points_ref, device, levels, squared_distances)
