"""Script that tests the compiled Cupy KDTree
"""
#import sys
#import os
#sys.path.append(os.path.dirname(__file__) + "/../") #TODO: Hack

import os
import pytest
import numpy as np
import cupy as cp
from cp_kdtree import build_kd_tree
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
  @pytest.mark.parametrize("device", ["cpu", "gpu"])
  @pytest.mark.parametrize("d", [1, 2, 3])
  @pytest.mark.parametrize("dtype", [np.float32, np.float64])
  def test_kdtree(self, nr_refs, nr_query, k, device, d, dtype):

    points_ref = np.random.uniform(size=(nr_refs, d)).astype(dtype) * 1e3
    points_query = np.random.uniform(size=(nr_query, d)).astype(dtype) * 1e3

    dists_ref, inds_ref = self.reference_solution(points_ref, points_query, k=k)
    cp_kdtree = build_kd_tree(points_ref, device=device)
    if device == 'gpu':
      points_query = cp.asarray(points_query)
    
    if k > nr_refs:
      with pytest.raises(Exception):
        dists, inds = cp_kdtree.query(points_query, nr_nns_searches=k)

      return #Correctly aborted the run
    else:
      dists, inds = cp_kdtree.query(points_query, nr_nns_searches=k)

    if device == 'gpu':
      assert(type(dists) == cp.ndarray)
      assert(type(inds) == cp.ndarray)
      dists = dists.get()
      inds = inds.get()
      points_query = points_query.get()

    #Shape checks
    assert(inds.shape[-1] == k)
    assert(inds.shape[0] == points_query.shape[0])
    assert(np.all(inds.shape == dists.shape))
    assert((dists_ref.ndim == 1 and dists.ndim == 2 and dists.shape[-1] == 1)
                    or np.all(dists_ref.shape == dists.shape))

    self.check_successful(points_ref, points_query, k, dists_ref, inds_ref, dists, inds)

  @pytest.mark.parametrize("device", ["gpu", "cpu"])
  def test_uncompiled_dimension(self, device):
    dims = 436
    points_ref = np.random.uniform(size=(10, dims)) 
    with pytest.raises(Exception):
      cp_kdtree = build_kd_tree(points_ref, device=device)

  @pytest.mark.parametrize("device", ["gpu", "cpu"])
  @pytest.mark.parametrize("dtype", [np.float32, np.float64])
  def test_wrong_dtype(self, device, dtype):
    dtypes = np.array([np.float32, np.float64])
    dims = 3
    points_ref = np.random.uniform(size=(10, dims))
    points_ref = points_ref.astype(dtype)
    cp_kdtree = build_kd_tree(points_ref, device=device)
    points_query = np.random.uniform(size=(10, dims)).astype(dtypes[np.array(dtype) != dtypes][0])
    with pytest.raises(Exception):
      result = cp_kdtree.query(points_query) #Call with the wrong dtype

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
