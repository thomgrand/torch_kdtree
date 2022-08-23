"""Script that tests the compiled Torch KDTree
"""
#import sys
#import os
#sys.path.append(os.path.dirname(__file__) + "/../") #TODO: Hack

import os
import pytest
import numpy as np
from torch_kdtree import build_kd_tree, gpu_available
import torch
import sys
import timeit
from scipy.spatial import cKDTree #Reference implementation

np.random.seed(0)

class TestCPKDTreeImplementation():

  #def __init__(self, test_name = None):
  #  super(TestCPKDTreeImplementation, self).__init__(test_name)


  def reference_solution(self, points_ref, points_query, k):
    kdtree = cKDTree(points_ref)
    dists, inds = kdtree.query(points_query, k)

    return dists, inds

  @pytest.mark.parametrize("nr_refs", [5, 10, 20, 30, 1000, 10000]) #, 100000])
  @pytest.mark.parametrize("nr_query", [5, 10, 20, 30 , 1000]) #, 10000]) #, 100000])
  @pytest.mark.parametrize("k", [1, 5, 10, 20]) #, 100])
  @pytest.mark.parametrize("device", ["cpu", "cuda"])
  @pytest.mark.parametrize("d", [1, 2, 3])
  @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
  @pytest.mark.parametrize("requires_grad", [True, False])
  def test_kdtree(self, nr_refs, nr_query, k, device, d, dtype, requires_grad):

    if device == "cuda" and not gpu_available:
      pytest.skip("No GPU")

    points_ref = torch.randn(size=(nr_refs, d), dtype=dtype, device=device) * 1e3
    points_query = torch.randn(size=(nr_query, d), dtype=dtype, device=device, requires_grad=requires_grad) * 1e3

    dists_ref, inds_ref = self.reference_solution(points_ref.detach().cpu().numpy(), points_query.detach().cpu().numpy(), k=k)
    t_kdtree = build_kd_tree(points_ref, device=torch.device(device))
    
    if k > nr_refs:
      with pytest.raises(Exception):
        dists, inds = t_kdtree.query(points_query, nr_nns_searches=k)

      return #Correctly aborted the run
    else:
      dists, inds = t_kdtree.query(points_query, nr_nns_searches=k)

    if device == 'cuda':
      assert dists.device.type == "cuda" and inds.device.type == "cuda"

    dists = dists.detach().cpu().numpy()
    points_query = points_query.detach().cpu().numpy()
    inds = inds.detach().cpu().numpy()

    #Shape checks
    assert(inds.shape[-1] == k)
    assert(inds.shape[0] == points_query.shape[0])
    assert(np.all(inds.shape == dists.shape))
    assert((dists_ref.ndim == 1 and dists.ndim == 2 and dists.shape[-1] == 1)
                    or np.all(dists_ref.shape == dists.shape))

    self.check_successful(points_ref.detach().cpu().numpy(), points_query, k, dists_ref, inds_ref, dists, inds)

  @pytest.mark.parametrize("device", ["cuda", "cpu"])
  def test_uncompiled_dimension(self, device):
    dims = 436
    points_ref = np.random.uniform(size=(10, dims)) 
    with pytest.raises(Exception):
      t_kdtree = build_kd_tree(points_ref, device=device)

  @pytest.mark.parametrize("device", ["cuda", "cpu"])
  @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
  def test_wrong_dtype(self, device, dtype):
    dtypes = np.array([torch.float32, torch.float64])
    dims = 3
    points_ref = torch.randn(size=(10, dims), dtype=dtype)
    t_kdtree = build_kd_tree(points_ref, device=device)
    points_query = torch.randn(size=(10, dims), dtype=dtypes[np.array(dtype) != dtypes][0])
    with pytest.raises(Exception):
      result = t_kdtree.query(points_query) #Call with the wrong dtype

  @pytest.mark.parametrize("device", ["cuda", "cpu"])
  @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
  def test_grad(self, device, dtype):
    dims = 3
    points_ref = torch.randn(size=(10, dims), dtype=dtype, device=device, requires_grad=True)
    t_kdtree = build_kd_tree(points_ref)
    points_query = torch.randn(size=(5, dims), dtype=dtype, device=device, requires_grad=True)
    dists, inds = t_kdtree.query(points_query, nr_nns_searches=2)
    (0.5 * torch.sum(dists)).backward()

    assert points_query.grad is not None
    assert torch.allclose(points_query.grad, torch.sum((points_query[:, None] - points_ref[inds]), axis=-2))
    assert points_ref.grad is not None
    points_ref_grad = torch.zeros_like(points_ref)
    for i, ind in enumerate(inds):
      points_ref_grad[ind] += points_ref[ind] - points_query[i]

    assert torch.allclose(points_ref.grad, points_ref_grad)



  def check_successful(self, points_ref, points_query, k, dists_ref, inds_ref, dists_knn, inds_knn):

    if dists_ref.ndim == 1:
      #dists_knn = dists_knn[..., 0]
      #inds_knn = inds_knn[..., 0]
      dists_ref = dists_ref[..., np.newaxis]
      inds_ref = inds_ref[..., np.newaxis]

    assert(
      np.allclose(dists_ref ** 2, np.sum((points_query[:, np.newaxis] - points_ref[inds_ref]) ** 2, axis=-1),
                  atol=1e-5))
    assert(
      np.allclose(dists_knn, np.sum((points_query[:, np.newaxis] - points_ref[inds_knn]) ** 2, axis=-1), atol=1e-5))
    assert(
      np.allclose(dists_ref ** 2, np.sum((points_query[:, np.newaxis] - points_ref[inds_knn]) ** 2, axis=-1),
                  atol=1e-5))
    assert(
      np.allclose(dists_knn, np.sum((points_query[:, np.newaxis] - points_ref[inds_ref]) ** 2, axis=-1), atol=1e-5))

    assert np.allclose(dists_ref ** 2, dists_knn, atol=1e-5), "Mismatch in KNN-Distances"

    # For larger values this sometimes flip
    # if k <= 100 and nr_query < 1e5 and nr_refs < 1e5:
    #  assert(np.all(inds_ref == inds_knn), "Mismatch in KNN-Indices")
    # else:
    assert np.sum(inds_ref == inds_knn) / inds_ref.size > 0.95, "Too many mismatches in KNN-Indices"
