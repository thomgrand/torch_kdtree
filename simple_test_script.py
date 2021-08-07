from scipy.spatial import cKDTree
import cp_knn
import numpy as np
import cupy as cp
points = np.random.normal(size=(int(1e4), 3)).astype(np.float32)
points_query = np.random.normal(size=(100, 3)).astype(np.float32)

nr_nns_searches = 20
dists = np.empty(shape=[points_query.shape[0], nr_nns_searches], dtype=np.float32)
inds = np.empty(shape=[points_query.shape[0], nr_nns_searches], dtype=np.int32)
kdtree = cp_knn.KDTreeCPU3DF(points, levels=8)
shuffled_inds = kdtree.get_shuffled_inds()
kdtree.query(points_query.__array_interface__['data'][0], 
            points_query.shape[0], 
            nr_nns_searches,
            dists.__array_interface__['data'][0],
            inds.__array_interface__['data'][0])
inds = shuffled_inds[inds]

dists_gpu = cp.empty(shape=[points_query.shape[0], nr_nns_searches], dtype=np.float32)
inds_gpu = cp.empty(shape=[points_query.shape[0], nr_nns_searches], dtype=np.int32)
kdtree_gpu = cp_knn.KDTreeGPU3DF(points, levels=8)
shuffled_inds_gpu = cp.asarray(kdtree_gpu.get_shuffled_inds())
points_query_gpu = cp.asarray(points_query)
kdtree_gpu.query(points_query_gpu.data.ptr, 
            points_query_gpu.shape[0], 
            nr_nns_searches,
            dists_gpu.data.ptr,
            inds_gpu.data.ptr)
inds_gpu = shuffled_inds_gpu[inds_gpu]

#Reference implementation
kdtree_ref = cKDTree(points)
dists_ref, inds_ref = kdtree_ref.query(points_query, k=nr_nns_searches)

assert(np.all(inds_ref == inds))
assert(np.allclose(dists_ref, np.sqrt(dists)))
assert(np.all(inds_ref == inds_gpu.get()))
assert(np.allclose(dists_ref, np.sqrt(dists_gpu.get())))

def tmp():
    kdtree_gpu.query(points_query_gpu.data.ptr, 
            points_query_gpu.shape[0], 
            nr_nns_searches,
            dists_gpu.data.ptr,
            inds_gpu.data.ptr)
    cp.cuda.runtime.deviceSynchronize()


#%timeit dists_ref, inds_ref = kdtree_ref.query(points_query, k=nr_nns_searches)
#%timeit kdtree.query(points_query.__array_interface__['data'][0],             points_query.shape[0],             nr_nns_searches,            dists.__array_interface__['data'][0],            inds.__array_interface__['data'][0])
#%timeit tmp()