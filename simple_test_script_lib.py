from scipy.spatial import cKDTree
from nn_distance import build_kd_tree
import numpy as np
import cupy as cp
points = np.random.normal(size=(int(1e4), 3)).astype(np.float32)
points_query = np.random.normal(size=(100, 3)).astype(np.float32)

nr_nns_searches = 20
kdtree = build_kd_tree(points, device='cpu')
dists, inds = kdtree.query(points_query, nr_nns_searches)

kdtree_gpu = build_kd_tree(points, device='gpu')
dists_gpu, inds_gpu = kdtree_gpu.query(points_query, nr_nns_searches)

#Reference implementation
kdtree_ref = cKDTree(points)
dists_ref, inds_ref = kdtree_ref.query(points_query, k=nr_nns_searches)

assert(np.all(inds_ref == inds))
assert(np.allclose(dists_ref, np.sqrt(dists)))
assert(np.all(inds_ref == inds_gpu.get()))
assert(np.allclose(dists_ref, np.sqrt(dists_gpu.get())))

def tmp():
    dists_gpu, inds_gpu = kdtree_gpu.query(points_query, nr_nns_searches)
    cp.cuda.runtime.deviceSynchronize()


%timeit dists_ref, inds_ref = kdtree_ref.query(points_query, k=nr_nns_searches)
%timeit kdtree.query(points_query, nr_nns_searches)
%timeit tmp()