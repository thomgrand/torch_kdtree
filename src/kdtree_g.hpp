#pragma once

const int max_leaves = 1024;
const int max_partitions = max_leaves-1;
const int nr_buffered_query_points = 1024;
const int nr_buffered_leaf_inds = 256;

#ifndef __CUDACC__

/**
 * @brief Ternary (cond ? a : b) helper to decide at compile time which type will be returned.
 *        Will return the side which types comply with T
 * 
 * @tparam T Return type
 * @tparam comp1 lhs type
 * @tparam comp2 rhs type
 * @return lhs if type(T) == lhs, else rhs
 */
template <typename T, typename comp1, typename comp2>
inline T& ternaryHelper(comp1& lhs, comp2& rhs)
{
    if constexpr (std::is_same<T, comp1>::value)
        return lhs;

    if constexpr (std::is_same<T, comp2>::value)
        return rhs;

    throw std::exception();
}

/**
 * @brief Copies all partitions and the associated KD-Tree data
 * 
 * @tparam T Type of the points
 * @tparam dims Dimensionality of the points
 * @param partition_info Reference to the partition info
 * @return PartitionInfoDevice<T, dims>* The copied partition info in the device memory
 */
template <typename T, dim_t dims>
PartitionInfoDevice<T, dims>* copyPartitionToGPU(const PartitionInfo<T, dims>& partition_info);

/**
 * @brief Deletes the partition info and the associated KD-Tree data from the device
 * 
 * @tparam T Type of the points
 * @tparam dims Dimensionality of the points
 * @param partition_info Pointer to the partition info on the device
 */
template <typename T, dim_t dims>
void freePartitionFromGPU(PartitionInfoDevice<T, dims>* partition_info);

/**
 * @brief Deletes the partition info and the associated KD-Tree data from the device
 * 
 * @tparam T Type of the points
 * @tparam T_calc Type of the calculations
 * @tparam dims Dimensionality of the points
 * @param partition_info Pointer to the partition info on the device
 * @param nr_query Number of points to be queried from the tree
 * @param points_query Points to be queried from the tree
 * @param dist Pointer to a distance array on the devices with nr_query * nr_nns_searches elements, where the knn squared distances will be found after the call
 * @param idx Pointer to an index array on the devices with nr_query * nr_nns_searches elements, where the knn indices will be found after the call
 * @param nr_nns_searches Number of nearest neighbors to query (=k)
 */
template
<typename T, typename T_calc, dim_t dims>
void KDTreeKNNGPUSearch(PartitionInfoDevice<T, dims>* partition_info,
                    const point_i_t nr_query, 
                    const std::array<T, dims>* points_query, T * dist, point_i_t* idx, const point_i_knn_t nr_nns_searches);
#endif