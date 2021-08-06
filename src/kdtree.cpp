#include <cstdint>
#include <stdint.h>
#include "kdtree.hpp"
#include "nndistance.hpp"

template
<typename T, typename T_calc, dim_t dims>
void compQuadrDistLeafPartition(const std::array<T, dims>& point, const PartitionLeaf<T, dims>& partition_leaf,
                                    std::vector<T>& best_dists, std::vector<point_i_knn_t>& best_idx,
									const point_i_knn_t nr_nns_searches)
{
	/*const std::array<T, dims>* partition_data = partition_leaf.data;
    const point_i_t partition_size = partition_leaf.nr_points;
	const point_i_t partition_offset = partition_leaf.offset;
    for(point_i_t ref_i = 0; ref_i < partition_size; ref_i++)
    {
        const T_calc dist = compQuadrDist<T, T_calc, dims>(point, partition_data[ref_i]);
        const auto insertion_idx = knnInsertionDynamic<T_calc>(dist, best_dists.data(), nr_nns_searches);
        if(insertion_idx < nr_nns_searches)
        {
            //Shift elements to the right
            //std::cout << k << ", " << d << "insert into " << insertion_idx << std::endl;
            std::move_backward(best_dists.begin() + insertion_idx, best_dists.end() - 1, best_dists.end());
            std::move_backward(best_idx.begin() + insertion_idx, best_idx.end() - 1, best_idx.end());
            best_dists[insertion_idx] = dist;
            best_idx[insertion_idx] = ref_i + partition_offset;
        }
    }  */
    compQuadrDistLeafPartition<T, T_calc, dims>(point, partition_leaf, best_dists.data(), best_idx.data(), nr_nns_searches);
}


template
<typename T, typename T_calc, dim_t dims>
void searchKDTreeIteratively(const std::array<T, dims>& point, 
                        PartitionInfo<T, dims>& partition_info,
                        std::vector<T>& best_dists, std::vector<point_i_knn_t>& best_idx,
                        const point_i_knn_t nr_nns_searches)
{
    TreeTraversal<T, dims> tree(&partition_info);
    std::array<T, dims> point_proj(point);
    auto worst_dist = best_dists.back();
    const tree_visit_f<T> func = [&tree, &point, &point_proj, &worst_dist](const Partition<T>& current_partition, 
    const NodeTag tag, const tree_ind_t current_lin_ind, const level_t current_level)
    {
        const auto current_axis = current_partition.axis_split;
        const bool lower_than_median = point[current_axis] < current_partition.median;

        //All branches of the node were visited
        if(tag == NodeTag::left_right_visited)
        {
            if(current_level == 0)
                return NodeDirection::finished; //Searched everything, quit
            else
            {
                point_proj[current_axis] = point[current_axis];
                return NodeDirection::up; //Search upper levels, nothing to do here
            }
        }

        //Arrived at uncomputed leaves
        if(tree.isLeafParent())
        {
            assert(tag == NodeTag::uncharted);
            tree.setCurrentNodeTag(NodeTag::left_right_visited);
            return NodeDirection::finished;
        }
        else
        {
            //Node has not yet been used
            if(tag == NodeTag::uncharted)
            {
                if(lower_than_median)
                {
                    tree.setCurrentNodeTag(NodeTag::left_visited);
                    return NodeDirection::left;
                }
                else
                {
                    tree.setCurrentNodeTag(NodeTag::right_visited);
                    return NodeDirection::right;
                }
            }
            else //Node has been used in the correct side, now we use the non-matching site: Project and test if descent is necessary
            {
                assert(!lower_than_median || tag == NodeTag::left_visited);
                assert(lower_than_median || tag == NodeTag::right_visited);
                tree.setCurrentNodeTag(NodeTag::left_right_visited); //Either way we finish all nodes here
                point_proj[current_axis] = current_partition.median;
                if(partitionNecessary<T, T_calc, dims>(point, point_proj, current_partition, worst_dist))       
                    return lower_than_median ? NodeDirection::right : NodeDirection::left;
                else
                {
                    point_proj[current_axis] = point[current_axis];
                    return (current_level != 0 ? NodeDirection::up : NodeDirection::finished);
                }
                
            }
            
        }
    };

    tree.traverseTree(func);
    do
    {
        assert(tree.getCurrentLevel() == tree.getTotalLevels() - 1);
        assert(tree.isLeafParent());

        const auto& current_partition = tree.getCurrentConstPartition();
        const auto current_axis = current_partition.axis_split;

        const bool lower_than_median = point[current_axis] < current_partition.median;
        const PartitionLeaf<T, dims>& partition_leaf = (lower_than_median ? 
                                                tree.getLeftLeaf() : tree.getRightLeaf());
        compQuadrDistLeafPartition<T, T_calc, dims>(point, partition_leaf, best_dists, best_idx, nr_nns_searches);
        worst_dist = best_dists.back();
        if(partitionNecessary<T, T_calc, dims>(point, point_proj, current_partition, worst_dist))
        {
            const PartitionLeaf<T, dims>& partition_leaf = (!lower_than_median ? 
                                                tree.getLeftLeaf() : tree.getRightLeaf());
            compQuadrDistLeafPartition<T, T_calc, dims>(point, partition_leaf, best_dists, best_idx, nr_nns_searches);
        }

        worst_dist = best_dists.back();
        tree.traverseTree(func);
    }while(tree.getCurrentLevel() != 0);
}

template
<typename T, typename T_calc, dim_t dims>
void compQuadrDistNode(const std::array<T, dims>& point, const Partition<T>* partitions, const PartitionLeaf<T, dims>* leaves,
                                const tree_ind_t lin_ind,
                                    std::vector<T>& best_dists, std::vector<point_i_knn_t>& best_idx,
                                    const int current_level, const std::array<T, dims>& point_proj, const point_i_knn_t nr_nns_searches)
{
    const auto& partition = partitions[lin_ind];
    dim_t current_axis = partition.axis_split;
    const bool lower_than_median = point[current_axis] < partition.median;

    if(current_level == 0)
    {
        const PartitionLeaf<T, dims>& partition_leaf_first = (lower_than_median ? 
                                                leaves[TreeTraversal<T, dims>::compLeftLeafInd(lin_ind)] : 
                                                leaves[TreeTraversal<T, dims>::compRightLeafInd(lin_ind)]);
        compQuadrDistLeafPartition<T, T_calc, dims>(point, partition_leaf_first, best_dists, best_idx, nr_nns_searches);
        if(partitionNecessary<T, T_calc, dims>(point, point_proj, partition, best_dists.back()))
        {
            const PartitionLeaf<T, dims>& partition_leaf_second = (!lower_than_median ? 
                                                leaves[TreeTraversal<T, dims>::compLeftLeafInd(lin_ind)] : 
                                                leaves[TreeTraversal<T, dims>::compRightLeafInd(lin_ind)]);
            compQuadrDistLeafPartition<T, T_calc, dims>(point, partition_leaf_second, best_dists, best_idx, nr_nns_searches);
        }
    }
    else
    {
        const tree_ind_t left_child_ind = TreeTraversal<T, dims>::compLeftChildInd(lin_ind), right_child_ind = TreeTraversal<T, dims>::compRightChildInd(lin_ind);
        compQuadrDistNode<T, T_calc, dims>(point, partitions, leaves, lower_than_median ? left_child_ind : right_child_ind, 
					best_dists, best_idx, current_level-1, point_proj, nr_nns_searches);

        if(partitionNecessary<T, T_calc, dims>(point, point_proj, partition, best_dists.back()))
        {
            std::array<T, dims> point_proj_new(point);
            point_proj_new[current_axis] = partition.median;
            compQuadrDistNode<T, T_calc, dims>(point, partitions, leaves, !lower_than_median ? left_child_ind : right_child_ind,
            best_dists, best_idx, current_level-1,
                               point_proj_new, nr_nns_searches);
        }
    }    
}

template
<typename T, typename T_calc, dim_t dims>
void compQuadrDistNode(const std::array<T, dims>& point, const std::vector<Partition<T>>& partitions, const std::vector<PartitionLeaf<T, dims>>& leaves,
                                const tree_ind_t lin_ind,
                                    std::vector<T>& best_dists, std::vector<point_i_knn_t>& best_idx,
                                    const int current_level, const std::array<T, dims>& point_proj, const point_i_knn_t nr_nns_searches)
{
    compQuadrDistNode<T, T_calc, dims>(point, partitions, leaves, lin_ind, best_dists, best_idx, current_level, point_proj, nr_nns_searches);
}

template
<typename T, typename T_calc, dim_t dims>
void KDTreeKNNSearch(PartitionInfo<T, dims>& partition_info,
                    const point_i_t nr_query, 
                    const std::array<T, dims>* points_query, T * dist, point_i_knn_t* idx, const point_i_knn_t nr_nns_searches)
{
    const auto levels = partition_info.levels;
    const auto& partitions = partition_info.partitions;
    const auto& leaves = partition_info.leaves;
    #pragma omp parallel for
	for (point_i_t query_i = 0; query_i < nr_query; query_i++)
    {
        const std::array<T, dims> query_point = points_query[query_i];
		std::vector<T_calc> best_dists(nr_nns_searches, std::numeric_limits<T_calc>::infinity());
		std::vector<point_i_knn_t> best_i(nr_nns_searches, 0);
    
        searchKDTreeIteratively<T, T_calc, dims>(query_point, partition_info, best_dists, best_i, nr_nns_searches);

        #ifndef NDEBUG
        std::vector<T_calc> best_dists2(nr_nns_searches, std::numeric_limits<T_calc>::infinity());
		std::vector<point_i_knn_t> best_i2(nr_nns_searches, 0);
        compQuadrDistNode<T, T_calc, dims>(query_point, partitions, leaves,  0, best_dists2, best_i2, levels-1, query_point, nr_nns_searches);

        assert(std::equal(best_dists.begin(), best_dists.end(), best_dists2.begin()));
        assert(std::equal(best_i.begin(), best_i.end(), best_i2.begin()));
        #endif

		std::copy(best_dists.begin(), best_dists.end(), dist + query_i*nr_nns_searches);
		std::copy(best_i.begin(), best_i.end(), idx + query_i*nr_nns_searches); //TODO: Type mismatch
	}
}

template void KDTreeKNNSearch<float, float, 3>(PartitionInfo<float, 3>& partition_info,
                    const point_i_t nr_query, 
                    const std::array<float, 3>* points_query, float * dist, point_i_t* idx, const point_i_knn_t nr_nns_searches);

template void KDTreeKNNSearch<double, double, 3>(PartitionInfo<double, 3>& partition_info,
                    const point_i_t nr_query, 
                    const std::array<double, 3>* points_query, double * dist, point_i_t* idx, const point_i_knn_t nr_nns_searches);

template void KDTreeKNNSearch<float, float, 2>(PartitionInfo<float, 2>& partition_info,
                    const point_i_t nr_query, 
                    const std::array<float, 2>* points_query, float * dist, point_i_t* idx, const point_i_knn_t nr_nns_searches);

template void KDTreeKNNSearch<double, double, 2>(PartitionInfo<double, 2>& partition_info,
                    const point_i_t nr_query, 
                    const std::array<double, 2>* points_query, double * dist, point_i_t* idx, const point_i_knn_t nr_nns_searches);

template void KDTreeKNNSearch<float, float, 1>(PartitionInfo<float, 1>& partition_info,
                    const point_i_t nr_query, 
                    const std::array<float, 1>* points_query, float * dist, point_i_t* idx, const point_i_knn_t nr_nns_searches);

template void KDTreeKNNSearch<double, double, 1>(PartitionInfo<double, 1>& partition_info,
                    const point_i_t nr_query, 
                    const std::array<double, 1>* points_query, double * dist, point_i_t* idx, const point_i_knn_t nr_nns_searches);

template <typename T, dim_t dims, bool delete_partitions>
PartitionInfo<T, dims, delete_partitions>::PartitionInfo(std::vector<Partition<T>>&& parts, std::vector<PartitionLeaf<T, dims>>&& leaves_, point_i_t* shuffled_inds_, const point_i_t nr_points_) : 
/*partitions(std::move(parts)), leaves(std::move(leaves_)),*/ shuffled_inds(shuffled_inds_), nr_points(nr_points_), 
nr_partitions(parts.size()), nr_leaves(leaves_.size())
{
    partitions = new Partition<T>[parts.size()]; std::move(parts.begin(), parts.end(), partitions);
    leaves = new PartitionLeaf<T, dims>[leaves_.size()]; std::move(leaves_.begin(), leaves_.end(), leaves);
    levels = TreeTraversal<T, dims>::compLevel(nr_partitions-1);
    structured_points = leaves[0].data;
}

template struct PartitionInfo<float, 1, false>;
template struct PartitionInfo<float, 2, false>;
template struct PartitionInfo<float, 3, false>;
template struct PartitionInfo<double, 1, false>;
template struct PartitionInfo<double, 2, false>;
template struct PartitionInfo<double, 3, false>;

template struct PartitionInfo<float, 1, true>;
template struct PartitionInfo<float, 2, true>;
template struct PartitionInfo<float, 3, true>;
template struct PartitionInfo<double, 1, true>;
template struct PartitionInfo<double, 2, true>;
template struct PartitionInfo<double, 3, true>;
