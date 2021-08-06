from scipy.spatial import cKDTree
import cp_knn
import numpy as np
points = np.random.normal(size=(int(1e3), 3)).astype(np.float32)
points_query = np.random.normal(size=(100, 3)).astype(np.float32)


dists = np.empty(shape=[points_query.shape[0], 20], dtype=np.float32)
inds = np.empty(shape=[points_query.shape[0], 20], dtype=np.int32)
kdtree = cp_knn.KDTreeCPU3DF(points, levels=5)
shuffled_inds = kdtree.get_shuffled_inds()
kdtree.query(points_query.__array_interface__['data'][0], 
            points_query.shape[0], 
            20,
            dists.__array_interface__['data'][0],
            inds.__array_interface__['data'][0])
inds = shuffled_inds[inds]
"""
kdtree_info = cp_knn.buildKDTreeCPUF(points, levels=3)
part_nr = kdtree_info[1]
shuffled_inds = kdtree_info[-1]
del kdtree_info

assert(part_nr == 0)
dists, inds = cp_knn.searchKDTreeCPUFA(points_query, 20, 0)
inds = shuffled_inds[inds]
"""
kdtree_ref = cKDTree(points)
dists_ref, inds_ref = kdtree_ref.query(points_query, k=20)

assert(np.all(inds_ref == inds))
assert(np.allclose(dists_ref, np.sqrt(dists)))


