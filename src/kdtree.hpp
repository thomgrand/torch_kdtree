#pragma once
/**
 * @file kdtree.hpp
 * @author Thomas Grandits
 * @brief Header of the KD-Tree implementation (CPU & GPU)
 * @version 0.1
 * @date 2020-12-16
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#include <iostream>     // std::cout
#include <iterator>     // std::back_inserter
#include <vector>       // std::vector
#include <algorithm>    // std::copy
#include <array>
#include <assert.h>     /* assert */
#include <cmath>
#include <tuple>
#include <numeric>
#include <functional>
#include "nndistance.hpp"

#include <Eigen/Dense>

//https://stackoverflow.com/questions/32014839/how-to-use-a-cuda-class-header-file-in-both-cpp-and-cuda-modules
#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

typedef uint8_t dim_t;
typedef int32_t point_i_t; //some values are signed for openmp support
typedef int32_t point_i_knn_t;
typedef point_i_t tree_ind_t;
typedef int level_t;
typedef uint8_t slot_t;

template <typename T, dim_t dims>
using Vec = Eigen::Matrix<T, dims, 1>;

/**
 * @brief PingPongBuffer consisting of two actual buffers that can be easily accessed and swapped
 * 
 * @tparam T Type of the PingPongBuffer
 */
template <typename T>
struct PingPongBuffer
{
    //std::array<T*, 2> buffers;
    T* buffers[2];
    slot_t current_slot = 0;

    CUDA_HOSTDEV PingPongBuffer(){}
    CUDA_HOSTDEV PingPongBuffer(T* ping, T* pong){buffers[0] = ping; buffers[1] = pong;}

    CUDA_HOSTDEV inline T* getCurrentSlot() { return buffers[current_slot]; }
    CUDA_HOSTDEV inline slot_t getCurrentSlotInd() const { return current_slot; }
    CUDA_HOSTDEV inline T* getPongSlot() { return buffers[(current_slot + 1) % 2]; }
    CUDA_HOSTDEV inline void increment()  { current_slot = (current_slot + 1) % 2; }
    CUDA_HOSTDEV inline T* getCurrentSlotAndIncrement() { T* buf = getCurrentSlot();  increment(); return buf;}
};

/**
 * @brief Computes the total number of nodes necessary for a KD-Tree with the given levels
 * 
 * @tparam T Type of the levels
 * @param levels Number of levels of the tree
 * @return T Number of total nodes of the tree
 */
template <typename T>
inline T compTotalNrNodes(const T levels)
{
    std::vector<T> range(levels);
    std::iota(range.begin(), range.end(), 0);

    return std::accumulate(range.begin(), range.end(), 0, [](const T accum, const T cur_level){return accum + std::pow<T>(2, cur_level);});
}

/**
 * @brief Computes the total number of leaves necessary for a KD-Tree with the given levels
 * 
 * @tparam T Type of the levels
 * @param levels Number of levels of the tree
 * @return T Number of total leaves of the tree
 */
template <typename T>
CUDA_HOSTDEV inline T compTotalNrLeaves(const T levels)
{
    return std::pow<T>(2, levels);
}

/**
 * @brief Enum to tag which nodes have been visited when traversing the tree
 * 
 */
enum NodeTag : unsigned char
{
    uncharted = 0,
    //visited = 1,
    left_visited = 1,
    right_visited = 2,
    left_right_visited = left_visited | right_visited
};

/**
 * @brief Enum to specify in which direction an algorithm should continue traversing the tree
 * 
 */
enum NodeDirection : unsigned char
{
    up = 0,
    left = 1,
    right = 2,
    finished = 3
};

/**
 * @brief Specifies data associated to each leaf of the tree
 * 
 * @tparam T Type of the leaf
 * @tparam dims Dimensionality of the contained points
 */
template <typename T, dim_t dims>
struct PartitionLeaf
{
    std::array<T, dims>* data;
    point_i_t nr_points;
    point_i_t offset;

    PartitionLeaf(std::array<T, dims>* data_, const point_i_t nr_points_, const point_i_t offset_) 
    : data(data_), nr_points(nr_points_), offset(offset_){}

    PartitionLeaf(){}
};

/**
 * @brief Partition info present at each node, denoting the split-axis and the median used for the partitioning 
 * 
 * @tparam T Type of the median
 */
template <typename T>
struct Partition //: PartitionAligned<T>
{
    dim_t axis_split;
    T median;

    Partition(const dim_t axis_split_, const T median_) : axis_split(axis_split_), median(median_){}
    Partition(){}
};

template <typename T>
using tree_visit_f = std::function<NodeDirection(const Partition<T>&, const NodeTag, const tree_ind_t, const level_t)>;

/**
 * @brief Contains all partition info associated with a KD-Tree:
 *        All partitions, all leaves, levels of the tree and the underyling structured points, as well as the shuffled indices are all managed here.
 * 
 * @tparam T Type of the structured points and all underlying data
 * @tparam dims Dimensionality of the points
 * @tparam delete_partitions If true, deletes all associated partition data once the tree is destructed
 */
template <typename T, dim_t dims, bool delete_partitions = true>
struct PartitionInfo
{
    Partition<T>* partitions;
    PartitionLeaf<T, dims>* leaves;
    level_t levels;
    std::array<T, dims>* structured_points = NULL;
    point_i_t* shuffled_inds = NULL;
    const point_i_t nr_points = 0;
    const tree_ind_t nr_partitions, nr_leaves;

    PartitionInfo(std::vector<Partition<T>>&& parts, std::vector<PartitionLeaf<T, dims>>&& leaves_, point_i_t* shuffled_inds_, const point_i_t nr_points_);
    
    PartitionInfo(PartitionInfo&& to_move) : partitions(std::move(to_move.partitions)), leaves(std::move(to_move.leaves)), levels(to_move.levels), 
        structured_points(to_move.structured_points), shuffled_inds(to_move.shuffled_inds), nr_points(to_move.nr_points), 
        nr_partitions(to_move.nr_partitions), nr_leaves(to_move.nr_leaves)
    { 
        to_move.partitions = NULL;
        to_move.leaves = NULL;
        to_move.structured_points = NULL;
        to_move.shuffled_inds = NULL;
    }

    PartitionInfo(const PartitionInfo<T, dims>& to_copy) : partitions(to_copy.partitions), leaves(to_copy.leaves), levels(to_copy.levels), 
        structured_points(to_copy.structured_points), shuffled_inds(to_copy.shuffled_inds), nr_points(to_copy.nr_points), 
        nr_partitions(to_copy.nr_partitions), nr_leaves(to_copy.nr_leaves)
    {
        if(delete_partitions)
            throw std::runtime_error("Copies are not meant to do this here");
    }

    CUDA_HOSTDEV ~PartitionInfo()
    {
        if(delete_partitions)
        {
            delete partitions;
            delete leaves;
            delete structured_points;
            delete shuffled_inds;
        }
    }
};

#ifdef __CUDACC__
template <typename T_ind>
__device__ T_ind compRightChildIndD(const T_ind lin_ind) 
{ 
    return lin_ind * 2 + 2; 
}
#endif

template <typename T, dim_t dims>
using PartitionInfoDevice = PartitionInfo<T, dims, false>;

/**
 * @brief Helper class to effectively traverse the KD-tree without any recursions. 
 * 
 * @tparam T Type of the points
 * @tparam dims Dimensionality of the points
 */
template <typename T, dim_t dims>
class TreeTraversal
{
//protected:
public:
    PartitionInfo<T, dims>* partition_info = NULL;

    NodeTag* visited_info; 
    tree_ind_t current_lin_ind;
    level_t current_level;   
    point_i_t nr_nodes;

public:

    /**
     * @brief Resets all positions and tags, effectively reinitializing the object without the need to destruct it.
     * 
     * @return CUDA_HOSTDEV 
     */
    CUDA_HOSTDEV void resetPositionAndTags()
    {
        std::fill(visited_info, visited_info + nr_nodes, NodeTag::uncharted);
        current_lin_ind = current_level = 0;
    }

    TreeTraversal(PartitionInfo<T, dims>* partition_info_) : partition_info(partition_info_), nr_nodes(partition_info_->nr_partitions)
    {
        //visited_info = std::move(std::vector<NodeTag>(compTotalNrNodes(partition_info->levels), NodeTag::uncharted));
        visited_info = new NodeTag[nr_nodes];
        std::fill(visited_info, visited_info + nr_nodes, NodeTag::uncharted);
        resetPositionAndTags();
    }

    CUDA_HOSTDEV TreeTraversal(){}

    CUDA_HOSTDEV TreeTraversal(PartitionInfoDevice<T, dims>* partition_info_) : 
    //TODO: This could cause serious problems down the road!!!
    partition_info(reinterpret_cast<PartitionInfo<T, dims>*>(partition_info_)),
    //partition_info_d(partition_info_), 
    nr_nodes(partition_info_->nr_partitions)
    {
        //visited_info = std::move(std::vector<NodeTag>(compTotalNrNodes(partition_info->levels), NodeTag::uncharted));
        visited_info = new NodeTag[nr_nodes];
        std::fill(visited_info, visited_info + nr_nodes, NodeTag::uncharted);
        current_lin_ind = current_level = 0;
    }

    CUDA_HOSTDEV ~TreeTraversal()
    {
        delete visited_info;
    }

    template <typename T_ind>
    static CUDA_HOSTDEV T_ind compParentInd(const T_ind lin_ind) { return (lin_ind - 1)/2; }

    template <typename T_ind>
    static CUDA_HOSTDEV T_ind compLeftChildInd(const T_ind lin_ind) { return lin_ind * 2 + 1; }

    template <typename T_ind>
    static CUDA_HOSTDEV T_ind compRightChildInd(const T_ind lin_ind) { return lin_ind * 2 + 2; }

    template <typename T_ind>
    static CUDA_HOSTDEV T_ind compLevel(T_ind lin_ind)
    { 
        if(lin_ind == 0)
            return 1;

        T_ind level = 0; 
        while((lin_ind = compParentInd(lin_ind)) != 0)
            level++;

        return level + 2;
    }

    template <typename T_ind, typename T_ind2>
    static CUDA_HOSTDEV inline T_ind compLeftLeafInd(const T_ind lin_ind, const T_ind2 nr_partitions) { return compLeftChildInd(lin_ind) - nr_partitions; }

    template <typename T_ind, typename T_ind2>
    static CUDA_HOSTDEV inline T_ind compRightLeafInd(const T_ind lin_ind, const T_ind2 nr_partitions) { return compLeftLeafInd(lin_ind, nr_partitions) + 1; }

    template <typename T_ind>
    static CUDA_HOSTDEV inline T_ind compLeftLeafInd(const T_ind lin_ind) { return compLeftLeafInd(lin_ind, compTotalNrNodes(compLevel(lin_ind))); }

    template <typename T_ind>
    static CUDA_HOSTDEV inline T_ind compRightLeafInd(const T_ind lin_ind) { return compLeftLeafInd(lin_ind) + 1; }

    CUDA_HOSTDEV inline tree_ind_t compLeftLeafInd(){ return compLeftLeafInd(current_lin_ind); }
    CUDA_HOSTDEV inline tree_ind_t compRighLeafInd(){ return compRightLeafInd(current_lin_ind); }
    CUDA_HOSTDEV inline level_t getTotalLevels() const {return partition_info->levels;}
    CUDA_HOSTDEV inline tree_ind_t getCurrentLinearIndex() const {return current_lin_ind;}
    CUDA_HOSTDEV inline level_t getCurrentLevel() const {return current_level;}
    CUDA_HOSTDEV inline NodeTag getCurrentTag() const {return visited_info[current_lin_ind];}
    CUDA_HOSTDEV //Partition<T>& getCurrentPartition() {return partition_info->partitions[current_lin_ind];}
    CUDA_HOSTDEV inline const Partition<T>& getCurrentConstPartition() const {return partition_info->partitions[current_lin_ind];}
    CUDA_HOSTDEV inline void moveToParent(){ assert(current_level != 0 && current_lin_ind != 0); current_lin_ind = compParentInd(current_lin_ind); current_level -= 1; }
    CUDA_HOSTDEV inline void moveToLeftChild(){ assert(current_level < partition_info->levels);  current_lin_ind = compLeftChildInd(current_lin_ind); current_level += 1; }
    CUDA_HOSTDEV void moveToRightChild() { assert(current_level < partition_info->levels); current_lin_ind = compRightChildInd(current_lin_ind); current_level += 1; }
    CUDA_HOSTDEV inline bool isLeafParent() const { return current_level == partition_info->levels-1; }
    CUDA_HOSTDEV inline const PartitionLeaf<T, dims>& getLeftLeaf() const{ assert(isLeafParent()); return partition_info->leaves[compLeftLeafInd(current_lin_ind, partition_info->nr_partitions)]; }
    CUDA_HOSTDEV inline const PartitionLeaf<T, dims>& getRightLeaf() const{ assert(isLeafParent()); return partition_info->leaves[compRightLeafInd(current_lin_ind, partition_info->nr_partitions)]; }
    CUDA_HOSTDEV inline void setCurrentNodeTag(const NodeTag new_tag){ visited_info[current_lin_ind] = new_tag; }

    static_assert(NodeTag::left_right_visited == 3 && NodeTag::uncharted == 0, "The binary computation does not work without these values");
    CUDA_HOSTDEV inline NodeTag moveTagToBinaryPosition(const NodeTag tag) const 
    { 
        const unsigned char bit_idx  = current_lin_ind % 4;
        return static_cast<NodeTag>(tag << (bit_idx*2));
    }
    CUDA_HOSTDEV inline NodeTag getBinaryIndMask() const { return moveTagToBinaryPosition(NodeTag::left_right_visited);}
    CUDA_HOSTDEV inline NodeTag getCurrentTagBinary() const 
    {
        const auto byte_idx = current_lin_ind / 4;
        const unsigned char bit_idx  = current_lin_ind % 4;
        const unsigned char byte_data = visited_info[byte_idx];
        const unsigned char binary_bitmask = NodeTag::left_right_visited << (bit_idx*2);
        const unsigned char data = ( (byte_data & binary_bitmask) >> (bit_idx*2)) & NodeTag::left_right_visited;

        return static_cast<NodeTag>(data);
    }
    CUDA_HOSTDEV inline void setCurrentNodeTagBinary(const NodeTag new_bits)
    { 
        const auto byte_idx = current_lin_ind / 4;
        const NodeTag old_byte = visited_info[byte_idx];
        const NodeTag modified_mask = moveTagToBinaryPosition(NodeTag::left_right_visited); //getBinaryIndMask();
        const NodeTag unmodified_mask = static_cast<NodeTag>(~modified_mask);
        //const unsigned char bit_idx  = current_lin_ind % 4;
        visited_info[byte_idx] = static_cast<NodeTag>((unmodified_mask & old_byte) | (moveTagToBinaryPosition(new_bits) & modified_mask));
    }

    /**
     * @brief Traverses the tree with which the object was constructed and calls a function at each node.
     * 
     * @param direction_f Function to call at each node. This function will receive the current node along with some positional information.
     *                    The function has to return a direction in which the tree should be further traversed. Finished when the function 
     *                    returns NodeDirection::finished.
     * @return CUDA_HOSTDEV 
     */
    CUDA_HOSTDEV void traverseTree(const tree_visit_f<T>& direction_f)
    {
        NodeDirection next_dir;
        while((next_dir = direction_f(getCurrentConstPartition(), getCurrentTag(), current_lin_ind, current_level)) 
                        != NodeDirection::finished)
        {
            switch(next_dir)
            {
                case NodeDirection::up:
                    moveToParent(); break;
                case NodeDirection::left:
                    moveToLeftChild(); break;
                case NodeDirection::right:
                    moveToRightChild(); break;
            }
        }
    }
};

/**
 * @brief Computes the median along given axis for the ordered points
 * 
 * @tparam T Type of the points
 * @tparam dims Dimensionality of the points
 * @param points_ordered Points for which the median should be computed. Note that the points are required to be ordered
 *                       in the respective dimension. This should be ensured after the tree was already constructed.
 * @param nr_points Number of points present in the array
 * @param current_axis Axis for which the median should be computed
 * @return T The median
 */
template <typename T, dim_t dims>
inline T compMedian(const std::array<T, dims>* points_ordered, const point_i_t nr_points, const dim_t current_axis)
{
    assert(nr_points > 1);
    const point_i_t half_ind = nr_points / 2;
    if(nr_points % 2 == 1)
        return points_ordered[half_ind][current_axis];
    else
        return 0.5 * (points_ordered[half_ind][current_axis] + points_ordered[half_ind-1][current_axis]);
}

/**
 * @brief Creates the KD-Tree's partitions using a recursive algorithms
 * 
 * @tparam T Type of the points
 * @tparam dims Dimensionality of the points
 * @param partitions Vector of already generated (possibly empty) partitions, which will be initialized
 * @param leaves Empty vector of leaves that will be generated when creating the KD-tree
 * @param lin_ind Current node index (initially 0)
 * @param structured_points Pointer to the array of underlying ordered points
 * @param shuffled_inds Pointer to the array of underlying indices of the ordered points
 * @param nr_points Number of total points
 * @param levels Levels of the final KD-Tree
 * @param current_axis Current dimension in which we will split (initially 0)
 * @param arr_offset Current recursive offset of in the structured points array (initially 0)
 */
template <typename T, dim_t dims>
void createPartitionRecursive(
                                std::vector<Partition<T>>& partitions,
                                std::vector<PartitionLeaf<T, dims>>& leaves,
                                const tree_ind_t lin_ind,
                                std::array<T, dims>* structured_points, 
                                point_i_t* shuffled_inds, const point_i_t nr_points,
                                const int levels, const dim_t current_axis, const point_i_t arr_offset)
{
    typedef TreeTraversal<T, dims> tree_t;
    if(nr_points == 0)
        throw std::runtime_error("Error: Ran out of points while building KD-Tree. Either you required too many levels, or you used a lot of coplanar points");


    //Track global indices
    std::vector<point_i_t> idx(nr_points);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), 
    [current_axis, &structured_points](const point_i_t lhs, const point_i_t rhs)
    { return structured_points[lhs][current_axis] < structured_points[rhs][current_axis]; });

    //Reorder in the original
    //Work on copy and then copy back
    std::vector<std::array<T, dims>> structured_points_copy;
    std::vector<point_i_t> shuffle_copy(shuffled_inds, shuffled_inds + nr_points);
    for(size_t elem_i = 0; elem_i < nr_points; elem_i++)
        shuffled_inds[elem_i] = shuffle_copy[idx[elem_i]];

    
    std::sort(structured_points, structured_points + nr_points, 
    [current_axis](const std::array<T, dims>& lhs, const std::array<T, dims>& rhs){return lhs[current_axis] < rhs[current_axis];});

    const T median = compMedian<T, dims>(structured_points, nr_points, current_axis);

    std::array<T, dims>* points_lower = structured_points; 
    std::array<T, dims>* points_higher = structured_points + nr_points/2;
    const point_i_t nr_points_lower = nr_points/2;
    const point_i_t nr_points_higher = nr_points - nr_points_lower; //TODO: For duplicates, this may cause alterations...

    assert((nr_points - nr_points_lower) > 0);

    if(levels == 0)
    {        
        assert(tree_t::compLeftLeafInd(lin_ind) == leaves.size());
        leaves.push_back(std::move(PartitionLeaf<T, dims>(points_lower, nr_points_lower, arr_offset)));

        assert(tree_t::compRightLeafInd(lin_ind) == leaves.size());
        leaves.push_back(std::move(PartitionLeaf<T, dims>(points_higher, nr_points_higher, arr_offset + nr_points_lower)));
        partitions[lin_ind] = std::move(Partition<T>(current_axis, median));
    }
    else
    {
        createPartitionRecursive<T, dims>(
            partitions, leaves, TreeTraversal<T, dims>::compLeftChildInd(lin_ind),
            points_lower, shuffled_inds, nr_points_lower, levels-1, 
                (current_axis + 1) % dims, arr_offset);
        createPartitionRecursive<T, dims>(
            partitions, leaves, TreeTraversal<T, dims>::compRightChildInd(lin_ind),
            points_higher, shuffled_inds + nr_points_lower, nr_points_higher, levels-1, 
                (current_axis + 1) % dims, arr_offset + nr_points_lower);
        
        Partition<T> partition(current_axis, median);
        partitions[lin_ind] = std::move(partition);
    }
}

/**
 * @brief Recursively generates a KD-Tree using \ref createPartitionRecursive 'createPartitionRecursive'
 * 
 * @tparam T Type of the points
 * @tparam dims Dimensionality of the points
 * @param points_flat Array of points
 * @param nr_points Number of points
 * @param levels KD-Tree levels
 * @return PartitionInfo<T, dims> The PartitionInfo containing the information regarding the KD-Tree
 */
template <typename T, dim_t dims>
PartitionInfo<T, dims> createKDTree(const T* points_flat, 
                                        const point_i_t nr_points, const int levels)
{
    assert(levels >= 1);
    std::vector<PartitionLeaf<T, dims>> leaves; //(compTotalNrLeaves(levels));
    std::vector<Partition<T>> partitions(compTotalNrNodes(levels), Partition<T>(-1, -1));

    point_i_t* orig_inds = new point_i_t[nr_points];
    std::iota(orig_inds, orig_inds + nr_points, 0); //Initializing
    //std::cout << "Creating main partition: " << nr_points << ", " << levels << std::endl;
    const std::array<T, dims>* points = reinterpret_cast<const std::array<T, dims>*>(points_flat);
    //std::vector<std::array<T, dims>> structured_points(points, points + nr_points);
    std::array<T, dims>* structured_points = new std::array<T, dims>[nr_points];
    std::copy(points, points + nr_points, structured_points);
    createPartitionRecursive<T, dims>(partitions, leaves, 0,
                                        structured_points,
                                        orig_inds, nr_points, levels-1, 0, 0);
    //partitions[0] = new MainPartition<T, dims>(std::move(partitions[0]));
    /*MainPartition<T, dims> main_partition = MainPartition<T, dims>(std::move(
                        ),
                                        structured_points, orig_inds, nr_points, levels);*/

    //Sanity checks
    #ifndef NDEBUG
    point_i_t offset = 0;
    for(int i = 0; i < leaves.size(); i++)
    {
        assert(leaves[i].offset == offset);
        offset += leaves[i].nr_points;
    }
    for(int i = 1; i < leaves.size(); i++)
    {
        assert((leaves[i-1].data + leaves[i-1].nr_points) == leaves[i].data);
    }
    #endif

    assert(leaves.size() == compTotalNrLeaves(levels));

    return std::move(PartitionInfo<T, dims>(std::move(partitions), std::move(leaves), /*structured_points,*/ orig_inds, nr_points)); //std::move(main_partition);
}

/**
 * @brief Computes the difference between two points (lhs - rhs)
 * 
 * @tparam T Type of the points
 * @tparam T_calc Type of the resulting difference
 * @tparam dims Dimensionality of the points
 * @param lhs Array of points
 * @param rhs Array of points
 * @return std::array<T_calc, dims> The difference between the lhs and rhs
 */
template
<typename T, typename T_calc, dim_t dims>
inline CUDA_HOSTDEV std::array<T_calc, dims> compDiff(const std::array<T, dims>& lhs, const std::array<T, dims>& rhs)
{
    std::array<T_calc, dims> diffs;

    for(dim_t dim_i = 0; dim_i < dims; dim_i++)
    {
        diffs[dim_i] = static_cast<T_calc>(lhs[dim_i]) - static_cast<T_calc>(rhs[dim_i]);
    }

    return std::move(diffs);
}

/**
 * @brief Computes the sum of squared elem
 * 
 * @tparam T Type of the array
 * @tparam dims Number of points
 * @param x Array of values
 * @return T sum(x^2)
 */
template
<typename T, dim_t dims>
inline CUDA_HOSTDEV T compQuadrSum(const std::array<T, dims>& x)
{
    return std::accumulate(x.begin(), x.end(), 0., [](const T accum, const T elem){return accum + elem*elem;});
}

/**
 * @brief Computes the element-wise quadratic euclidean distance between lhs and rhs, i.e. ||lhs_i - rhs_i||
 * 
 * @tparam T Type of the array
 * @tparam T_calc Type of the calculation
 * @tparam dims Dimensionality of the points
 * @param lhs Left-hand-side of the calculation
 * @param lhs Right-hand-side of the calculation
 * @return T_calc ||lhs_i - rhs_i||
 */
template
<typename T, typename T_calc, dim_t dims>
inline CUDA_HOSTDEV T_calc compQuadrDist(const std::array<T, dims>& lhs, const std::array<T, dims>& rhs)
{
    std::array<T_calc, dims> diffs = compDiff<T, T_calc, dims>(lhs, rhs);
    return compQuadrSum<T_calc, dims>(diffs);
}

/**
 * @brief Computes the distance to the projected point onto the median of the current dimension. This is used to see if we need to further
 *        traverse the tree or already found the KNNs, which is true in case the projection is farther away than
 *        the distance to the current KNN most far away.
 * 
 * @tparam T Type of the array
 * @tparam dims Dimensionality of the points
 * @param point Original point for which to search the KNN
 * @param point_proj The original point, already projected on possible multiple previous dimensions. Note that in the simplest case
 *                   this is equal to point.
 * @param Partition<T> Node information of the current axis/dimension split that we want to compute the projection in
 * @return T Resulting distance of the current projection
 */
template
<typename T, dim_t dims>
inline CUDA_HOSTDEV T projectionDist(const std::array<T, dims>& point, const std::array<T, dims>& point_proj, const Partition<T>& partition)
{
    std::array<T, dims> proj_vec = compDiff<T, T, dims>(point_proj, point);
    const dim_t current_axis = partition.axis_split;
    proj_vec[current_axis] += partition.median - point_proj[current_axis];
    return compQuadrSum<T, dims>(proj_vec);
}

/**
 * @brief See \ref projectionDist 'projectionDist'
 */
template
<typename T, dim_t dims>
inline CUDA_HOSTDEV T projectionDist(const Vec<T, dims>& point, const Vec<T, dims>& point_proj, const Partition<T>& partition)
{
    Vec<T, dims> proj_vec = (point_proj - point);
    const dim_t current_axis = partition.axis_split;
    proj_vec[current_axis] += partition.median - point_proj[current_axis];
    return proj_vec.squaredNorm();
}

/**
 * @brief Checks if we need to compute the distances to any of the points in the partition. Achieved by using \ref projectionDist 'projectionDist'.
 */
template
<typename T, typename T_calc, dim_t dims>
inline CUDA_HOSTDEV bool partitionNecessary(const std::array<T, dims>& point, const std::array<T, dims>& point_proj, 
                                const Partition<T>& partition, const T current_worst_dist)
{
    const auto proj_dist = projectionDist<T, dims>(point, point_proj, partition);
    return proj_dist < current_worst_dist;
}

/**
 * @brief See \ref partitionNecessary 'partitionNecessary'.
 */
template
<typename T, typename T_calc, dim_t dims>
inline CUDA_HOSTDEV bool partitionNecessary(const Vec<T, dims>& point, const Vec<T, dims>& point_proj, 
                                const Partition<T>& partition, const T current_worst_dist)
{
    const auto proj_dist = projectionDist<T, dims>(point, point_proj, partition);
    return proj_dist < current_worst_dist;
}

/**
 * @brief Computes the quadratic distances between point to all points in partition_leaf and updates the current knn, marked by best_dists and best_idx. 
 * 
 * @param point Query point for which the KNN should be calculated
 * @param partition_leaf Partition for which all quadratic distances should be computed
 * @param best_dists Quadr. distances to the currently assumed KNNs (starts with infinity). Has a length of K
 * @param best_idx Indices of the currently assumed KNNs (starts with -1). Has a length of K
 * @param nr_nns_searches How many nearest neighbors will be searched (=K)
 *         
 */
template
<typename T, typename T_calc, dim_t dims>
void compQuadrDistLeafPartition(const std::array<T, dims>& point, const PartitionLeaf<T, dims>& partition_leaf,
                                    T* best_dists, point_i_knn_t* best_idx,
									const point_i_knn_t nr_nns_searches)
{
	const std::array<T, dims>* partition_data = partition_leaf.data;
    const point_i_t partition_size = partition_leaf.nr_points;
	const point_i_t partition_offset = partition_leaf.offset;
    for(point_i_t ref_i = 0; ref_i < partition_size; ref_i++)
    {
        const T_calc dist = compQuadrDist<T, T_calc, dims>(point, partition_data[ref_i]);
        const auto insertion_idx = knnInsertionDynamic<T_calc>(dist, best_dists, nr_nns_searches);
        if(insertion_idx < nr_nns_searches)
        {
            const auto best_dists_end = best_dists + nr_nns_searches;
            const auto best_idx_end = best_idx + nr_nns_searches;
            assert(best_dists + insertion_idx < best_dists_end);
            assert(best_idx + insertion_idx < best_idx_end);
            //Shift elements to the right
            //std::cout << k << ", " << d << "insert into " << insertion_idx << std::endl;
            //std::move_backward(best_dists + insertion_idx, best_dists +  nr_nns_searches - 1, best_dists.end());
            //std::move_backward(best_idx.begin() + insertion_idx, best_idx.end() - 1, best_idx.end());
            moveBackward(best_dists + insertion_idx, best_dists_end - 1, best_dists_end);
            moveBackward(best_idx + insertion_idx, best_idx_end - 1, best_idx_end);
            best_dists[insertion_idx] = dist;
            best_idx[insertion_idx] = ref_i + partition_offset;
        }
    }    
}


/**
 * @brief See \ref compQuadrDistLeafPartition 'compQuadrDistLeafPartition'.
 */
template
<typename T, typename T_calc, dim_t dims>
CUDA_HOSTDEV void compQuadrDistLeafPartition(const Vec<T, dims>& point, const PartitionLeaf<T, dims>& partition_leaf,
                                    T* best_dists, point_i_knn_t* best_idx,
									const point_i_knn_t nr_nns_searches)
{
    //printf("compQuadrDistLeafPartition: %x, ", partition_leaf.data);
    //printf("%d, ", partition_leaf.nr_points);
    //printf("%d\n", partition_leaf.offset);
	const Vec<T, dims>* partition_data = reinterpret_cast<Vec<T, dims>*>(partition_leaf.data);
    const point_i_t partition_size = partition_leaf.nr_points;
	const point_i_t partition_offset = partition_leaf.offset;
    for(point_i_t ref_i = 0; ref_i < partition_size; ref_i++)
    {
        const T_calc dist = (point - partition_data[ref_i]).squaredNorm();
        const auto insertion_idx = knnInsertionDynamic<T_calc>(dist, best_dists, nr_nns_searches);
        if(insertion_idx < nr_nns_searches)
        {
            const auto best_dists_end = best_dists + nr_nns_searches;
            const auto best_idx_end = best_idx + nr_nns_searches;
            assert(best_dists + insertion_idx < best_dists_end);
            assert(best_idx + insertion_idx < best_idx_end);
            //Shift elements to the right
            //std::cout << k << ", " << d << "insert into " << insertion_idx << std::endl;
            //std::move_backward(best_dists + insertion_idx, best_dists +  nr_nns_searches - 1, best_dists.end());
            //std::move_backward(best_idx.begin() + insertion_idx, best_idx.end() - 1, best_idx.end());
            moveBackward(best_dists + insertion_idx, best_dists_end - 1, best_dists_end);
            moveBackward(best_idx + insertion_idx, best_idx_end - 1, best_idx_end);
            best_dists[insertion_idx] = dist;
            best_idx[insertion_idx] = ref_i + partition_offset;
        }
    }    
}

/**
 * @brief See \ref compQuadrDistLeafPartition 'compQuadrDistLeafPartition'.
 */
template
<typename T, typename T_calc, dim_t dims>
void compQuadrDistPartition(const std::array<T, dims>& point, const PartitionLeaf<T, dims>& partition_leaf,
                                    std::vector<T>& best_dists, std::vector<point_i_knn_t>& best_idx,
									const point_i_knn_t nr_nns_searches);

/**
 * @brief Recursively traverses the node and calls compQuadrDistLeafPartition on the leaves.
 */
template
<typename T, typename T_calc, dim_t dims>
void compQuadrDistNode(const std::array<T, dims>& point, const Partition<T>& partition,
                                    std::vector<T>& best_dists, std::vector<point_i_knn_t>& best_idx,
                                    const int current_level, const std::array<T, dims>& point_proj, const point_i_knn_t nr_nns_searches);

/**
 * @brief Computes the KNN on the KD-Tree defined by partition_info. Iteratively calls compQuadrDistNode on the main node for each point in points_query.
 *        The resulting quadr. distances and indices of the KNNs will be stored in dist [M, K] in a C-ordering fashion.
 */
template
<typename T, typename T_calc, dim_t dims>
void KDTreeKNNSearch(PartitionInfo<T, dims>& partition_info,
                    const point_i_t nr_query, 
                    const std::array<T, dims>* points_query, T * dist, point_i_knn_t* idx, const point_i_knn_t nr_nns_searches);

