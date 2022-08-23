"""This script computes the data for the benchmark
"""
import numpy as np
import timeit

ks = np.array([1, 10, 100])
nr_refs = np.logspace(2, 6, num=7).astype(np.int32)
nr_queries = np.logspace(1, 6, num=7).astype(np.int32)

if __name__ == "__main__":
    #from IPython import get_ipython
    from scipy.spatial import cKDTree
    from cp_kdtree import build_kd_tree
    import cupy as cp

    #ipython = get_ipython()

    dims = 3

    timing_results = np.zeros(shape=[2, nr_refs.size, nr_queries.size, ks.size])

    for ref_i, nr_ref in enumerate(nr_refs):
        #Create data
        points_ref = np.random.uniform(size=(nr_ref, dims)).astype(np.float32) * 1e3

        #Build KD-Trees right here to save some time
        kdtree = cKDTree(points_ref)
        cp_kdtree = build_kd_tree(points_ref, levels=None)

        for query_i, nr_query in enumerate(nr_queries):
            points_query = np.random.uniform(size=(nr_query, dims)).astype(np.float32) * 1e3
            points_query_cp = cp.asarray(points_query)

            for k_i, k in enumerate(ks):        
                print("------- {}, {}, {} --------".format(ref_i, query_i, k_i))
                    
                #Scipy spatial implementation                
                #timing = ipython.run_line_magic("timeit", "-o kdtree.query(points_query, k)")
                timing = timeit.timeit(lambda: kdtree.query(points_query, k), number=5) / 5
                timing_results[0, ref_i, query_i, k_i] = timing #.average

                #Cupy KD-Tree implementation 
                #result_dists = cp.empty(shape=[nr_query, k], dtype=cp_kdtree.dtype)
                #result_idx = cp.empty(shape=[nr_query, k], dtype=cp_kdtree.dtype_idx)
                cp_kdtree.query(points_query, nr_nns_searches=k) #, result_dists=result_dists, result_idx=result_idx) #Dry run to compile all JIT kernels
                #timing = ipython.run_line_magic("timeit", "-o cp_kdtree.query(points_query, nr_nns_searches=k); cp.cuda.runtime.deviceSynchronize()")
                def f():
                    cp_kdtree.query(points_query, nr_nns_searches=k)
                    cp.cuda.runtime.deviceSynchronize()

                timing = timeit.timeit(f, number=5) / 5
                #timing = ipython.run_line_magic("timeit", "-o cp_kdtree.query(points_query, nr_nns_searches=k, result_dists=result_dists, result_idx=result_idx); cp.cuda.runtime.deviceSynchronize()")
                timing_results[1, ref_i, query_i, k_i] = timing #.average
                print(f"Ratio: {timing_results[0, ref_i, query_i, k_i]/timing_results[1, ref_i, query_i, k_i]:.1f}")


    np.savez_compressed("benchmark_results.npz", timing_results=timing_results)
