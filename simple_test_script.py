from scipy.spatial import cKDTree
import cp_knn
import numpy as np
points = np.random.normal(size=(int(1e3), 3)).astype(np.float32)
points_query = np.random.normal(size=(100, 3)).astype(np.float32)
kdtree_info = cp_knn.buildKDTreeCPUF(points, levels=3)

assert(kdtree_info[1] == 0)
dists, inds = cp_knn.searchKDTreeCPUFA(points_query, 20, 0)
inds = kdtree_info[-1][inds]

kdtree_ref = cKDTree(points)
dists_ref, inds_ref = kdtree_ref.query(points_query, k=20)

assert(np.all(inds_ref == inds))
assert(np.allclose(dists_ref, np.sqrt(dists)))
