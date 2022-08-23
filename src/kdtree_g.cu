#define EIGEN_USE_GPU
#include "kdtree.hpp"
#include "nndistance.hpp"
#include "cutils.cuh"

const int local_dist_buf_size = 256;

//TODO: Sort and break for possible additional speedup
template
<typename T, typename T_calc, dim_t dims>
__device__ void compQuadrDistLeafPartitionBlockwise(const Vec<T, dims>& point, const PartitionLeaf<T, dims>& partition_leaf,
									T* local_dist_buf,
                                    PingPongBuffer<T>& best_dist_pp, PingPongBuffer<point_i_t>& best_knn_pp,
									const point_i_knn_t nr_nns_searches, T& worst_dist)
{
	//2D indices
	const int block_size = blockDim.x * blockDim.y;
	const int tidx = threadIdx.y * blockDim.x + threadIdx.x;

    /*printf("compQuadrDistLeafPartition: %x, ", partition_leaf.data);
    printf("%d, ", partition_leaf.nr_points);
	printf("%d\n", partition_leaf.offset);*/
	const Vec<T, dims>* partition_data = reinterpret_cast<Vec<T, dims>*>(partition_leaf.data);
    const point_i_t partition_size = partition_leaf.nr_points;
	const point_i_t partition_offset = partition_leaf.offset;
	//assert(local_dist_buf_size >= partition_size);
	assert(partition_size > 0);

	const auto nr_buf_runs = (partition_size - 1) / local_dist_buf_size + 1;
	for(size_t buf_run_i = 0; buf_run_i < nr_buf_runs; buf_run_i++)
	{
		const auto remaining_partition_size = partition_size - (buf_run_i * local_dist_buf_size);
		const auto current_length = min(static_cast<int>(remaining_partition_size), local_dist_buf_size);
		compDists<T, dims>(point, partition_data + (buf_run_i * local_dist_buf_size),
				current_length,
				local_dist_buf);
		__syncthreads();
		for(point_i_t ref_i = 0; ref_i < current_length; ref_i++)
		{
			T* best_dist_cur = best_dist_pp.getCurrentSlot();
			point_i_t* best_knn_cur = best_knn_pp.getCurrentSlot();
			T* pong_dist = best_dist_pp.getPongSlot();
			point_i_t* pong_knn = best_knn_pp.getPongSlot();
			//worst_dist = best_dist_cur[nr_nns_searches - 1];
			const T_calc calc_dist = local_dist_buf[ref_i];

			if(calc_dist < worst_dist)
			{
				const auto insertion_idx = knnInsertionDynamic<T_calc, false>(calc_dist, best_dist_cur, nr_nns_searches);
				assert(insertion_idx < nr_nns_searches);

				assertEqual(worst_dist, best_dist_cur[nr_nns_searches - 1]); //assert(worst_dist == best_dist_cur[nr_nns_searches - 1]);
				insertAndShiftArrayRight<T, true>(best_dist_cur, pong_dist, nr_nns_searches, calc_dist, insertion_idx, worst_dist);
				insertAndShiftArrayRight<point_i_t>(best_knn_cur, pong_knn, nr_nns_searches, 
						ref_i + partition_offset + (buf_run_i * local_dist_buf_size), 
													insertion_idx);
				best_dist_pp.increment();
				best_knn_pp.increment();
			}
			__syncthreads();
		}
	}
    worst_dist = best_dist_pp.getCurrentSlot()[nr_nns_searches - 1];
}

template <typename T, dim_t dims>
PartitionInfoDevice<T, dims>* copyPartitionToGPU(const PartitionInfo<T, dims>& partition_info)
{
	std::array<T, dims>* structured_points_d = copyArrayToDevice(partition_info.structured_points, partition_info.nr_points);

	std::vector<PartitionLeaf<T, dims>> leaves_copy(partition_info.leaves, partition_info.leaves + partition_info.nr_leaves);
	for(auto& leaf : leaves_copy)
		leaf.data = structured_points_d + leaf.offset;

    const auto& partitions = partition_info.partitions;
    Partition<T>* partitions_d = copyArrayToDevice(partitions, partition_info.nr_partitions);

    const auto& leaves = partition_info.leaves;
    PartitionLeaf<T, dims>* leaves_d = copyArrayToDevice(leaves_copy.data(), partition_info.nr_leaves);


    
    point_i_t* shuffled_inds_d = copyArrayToDevice(partition_info.shuffled_inds, partition_info.nr_points);

    PartitionInfoDevice<T, dims> partition_info_tmp(partition_info);
    partition_info_tmp.partitions = partitions_d;
    partition_info_tmp.leaves = leaves_d;
    partition_info_tmp.levels = partition_info.levels;
    partition_info_tmp.structured_points = structured_points_d;
    partition_info_tmp.shuffled_inds = shuffled_inds_d;
    
    PartitionInfoDevice<T, dims>* partition_info_d;
    allocGPUMemory(&partition_info_d, sizeof(PartitionInfo<T, dims>));
    gpuErrchk(cudaMemcpy(partition_info_d, &partition_info_tmp, sizeof(PartitionInfo<T, dims>), cudaMemcpyHostToDevice));

    #ifndef NDEBUG
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
	#endif
    return partition_info_d;
}

//Just necessary for the C++ implementation. Tensorflow already has everything on the GPU
template <typename T, dim_t dims>
std::tuple<T*, point_i_t*, T*> copyData(const std::vector<T>& result_dists, const std::vector<point_i_t>& result_idx, 
    const std::vector<std::array<T, dims>>& points_query)
{
    T* result_dists_d = copyArrayToDevice(result_dists.data(), result_dists.size());
    point_i_t* result_idx_d = copyArrayToDevice(result_idx.data(), result_idx.size());
    T* points_query_d = reinterpret_cast<T*>(copyArrayToDevice(points_query.data(), points_query.size()));

    #ifndef NDEBUG
	gpuErrchk( cudaPeekAtLastError() );
	//gpuErrchk( cudaDeviceSynchronize() );
	#endif
    return std::make_tuple(result_dists_d, result_idx_d, points_query_d);
}

//Just necessary for the C++ implementation. Tensorflow already has everything on the GPU
template <typename T>
std::tuple<T*, point_i_t*> copyDataBackToHost(const T* result_dists, const point_i_knn_t* result_idx, const size_t nr_query, const uint32_t nr_nns_searches)
{
    T* result_dists_h = copyArrayToHost(result_dists, nr_query * nr_nns_searches);
    point_i_t* result_idx_h = copyArrayToHost(result_idx,  nr_query * nr_nns_searches);

    #ifndef NDEBUG
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
	#endif
    return std::make_tuple(result_dists_h, result_idx_h);
}

template std::tuple<float*, point_i_t*> copyDataBackToHost(const float* result_dists, const point_i_knn_t* result_idx, const size_t nr_query, const uint32_t nr_nns_searches);
template std::tuple<double*, point_i_t*> copyDataBackToHost(const double* result_dists, const point_i_knn_t* result_idx, const size_t nr_query, const uint32_t nr_nns_searches);

/**
 * @brief Frees the allocated GPU memory for a single KD-Tree
 * 
 * @tparam T precision type of the KD-Tree
 * @tparam dims Dimensionality of the KD-Tree (usually 3)
 * @param partition_info Pointer to the device memory holding the KD-Tree information
 */
template <typename T, dim_t dims>
void freePartitionFromGPU(PartitionInfoDevice<T, dims>* partition_info)
{
	PartitionInfoDevice<T, dims>* local = reinterpret_cast<PartitionInfoDevice<T, dims>*>(malloc(sizeof(PartitionInfoDevice<T, dims>)));
    gpuErrchk(cudaMemcpy(local, partition_info, sizeof(PartitionInfo<T, dims>), cudaMemcpyDeviceToHost));
    freeGPUMemory(local->partitions);
    freeGPUMemory(local->leaves);
    freeGPUMemory(local->structured_points);
    freeGPUMemory(local->shuffled_inds);
    freeGPUMemory(partition_info);

    #ifndef NDEBUG
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
	#endif

	free(local);
}

/**
 * @brief Frees the allocated GPU memory of a KD-Tree result (currently only used by test_kdtree.cpp)
 *
 * @tparam T Array Type
 * @param arr Array to free
 */
template <typename T>
void freeGPUArray(T* arr)
{
	freeGPUMemory(arr);

#ifndef NDEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

/**
 * @brief 
 * 
 * @tparam T 
 * @tparam T_calc 
 * @tparam dims 
 * @param tree 
 * @param point 
 * @param point_proj 
 * @param current_worst_dist 
 * @return NodeDirection Holds the direction in which the tree should be further traversed 
 */
template <typename T, typename T_calc, dim_t dims>
__device__ NodeDirection traverseTree(TreeTraversal<T, dims>& tree, 
	const Vec<T, dims>& point, Vec<T, dims>& point_proj, const T current_worst_dist)
{
	//printf("traverseTree begin\n");
	const NodeTag tag = tree.getCurrentTagBinary();
	//printf("Node Tag %d\n", tag);
	//printf("Lin Ind %d\n", tree.current_lin_ind);

	const auto& current_partition = tree.getCurrentConstPartition();
	//printf("Fetched partition %d/%f\n", current_partition.axis_split, current_partition.median);

	const auto current_level = tree.getCurrentLevel();
	const auto current_axis = current_partition.axis_split;
	const bool lower_than_median = point[current_axis] < current_partition.median;

	//printf("Selecting new leaf\n");

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
		tree.setCurrentNodeTagBinary(NodeTag::left_right_visited);
		return NodeDirection::finished;
	}
	else
	{
		//Node has not yet been used
		if(tag == NodeTag::uncharted)
		{
			if(lower_than_median)
			{
				tree.setCurrentNodeTagBinary(NodeTag::left_visited);
				return NodeDirection::left;
			}
			else
			{
				tree.setCurrentNodeTagBinary(NodeTag::right_visited);
				return NodeDirection::right;
			}
		}
		else //Node has been used in the correct side, now we use the non-matching site: Project and test if descent is necessary
		{
			assert(!lower_than_median || tag == NodeTag::left_visited);
			assert(lower_than_median || tag == NodeTag::right_visited);
			tree.setCurrentNodeTagBinary(NodeTag::left_right_visited); //Either way we finish all nodes here
			if(partitionNecessary<T, T_calc, dims>(point, point_proj, current_partition, current_worst_dist))                     
			{
				point_proj[current_axis] = current_partition.median;
				return lower_than_median ? NodeDirection::right : NodeDirection::left;
			}
			else
			{
				point_proj[current_axis] = point[current_axis];
				return (current_level != 0 ? NodeDirection::up : NodeDirection::finished);
			}
			
		}
		
	}
}

template <typename T, typename T_calc, dim_t dims>
inline __device__ void findNextLeaf(TreeTraversal<T, dims>& tree, 
	const Vec<T, dims>& point, Vec<T, dims>& point_proj, const T current_worst_dist)
{
	NodeDirection new_dir;
	while((new_dir = traverseTree<T, T_calc, dims>(tree, point, point_proj, current_worst_dist)) != NodeDirection::finished)
	{
		switch(new_dir)
		{
			case NodeDirection::up:
				tree.moveToParent(); break;
			case NodeDirection::left:
				tree.moveToLeftChild(); break;
			case NodeDirection::right:
				tree.moveToRightChild(); break;
		}
	}
}

const int max_nr_nodes = 2048*4;
static_assert(max_nr_nodes % 4 == 0, "Alignment off, since 4 nodes fit into a byte");
const int max_nr_nns_searches = 128;

template 
<typename T, typename T_calc, dim_t dims>
__global__ void KDTreeKernel(PartitionInfoDevice<T, dims>* partition_info,
	const point_i_t nr_query, 
	const Vec<T, dims>* points_query, T* all_best_dists_d, point_i_knn_t* all_best_i_d, const point_i_knn_t nr_nns_searches)
{
	assert(nr_nns_searches <= partition_info->nr_points);

	const auto nr_partitions = partition_info->nr_partitions;
	const auto nr_leaves = partition_info->nr_leaves;
	//printf("%d, %d\n", nr_partitions, nr_leaves);

	//2D indices
	const auto grid_size = gridDim.x * gridDim.y;
	const auto blockidx = blockIdx.x + blockIdx.y*gridDim.x;
	const auto block_size = blockDim.x * blockDim.y; // * blockDim.z;
	const auto tidx = threadIdx.y * blockDim.x + threadIdx.x;
	//const auto global_start_idx = tidx + blockidx * grid_size;

	//extern __shared__ char* shared_mem;
	//__shared__ Vec<T, dims> buffered_query_points[nr_buffered_query_points];
	//__shared__ Vec<T, dims> buffered_query_points_proj[nr_buffered_query_points];
	//__shared__ tree_ind_t leaf_inds[nr_buffered_leaf_inds];
	__shared__ point_i_t buffered_knn[2*max_nr_nns_searches];
	__shared__ T buffered_dists[2*max_nr_nns_searches];
	__shared__ TreeTraversal<T, dims> tree[1];
	__shared__ NodeTag tags[max_nr_nodes/4];
	PingPongBuffer<T> best_dist_pp[1];
	PingPongBuffer<point_i_t> best_knn_pp;
	__shared__ T local_dist_buf[local_dist_buf_size];
	__shared__ Vec<T, dims> point_proj[1];
	__shared__ T worst_dist_[1];
	__shared__ bool off_leaf_necessary[1]; //TODO: Watch out for alignment
	
	//if(tidx == 0)
	//	local_dist_buf_pointer[0] = new T[local_dist_buf_size]; //TODO: Dynamic

	const auto nr_nodes = partition_info->nr_partitions;
	assert(max_nr_nodes >= nr_nodes);
	assert(nr_nns_searches <= max_nr_nns_searches);

	tree->partition_info = reinterpret_cast<PartitionInfo<T, dims>*>(partition_info);
	tree->visited_info = tags; //new NodeTag[tree->partition_info->nr_nodes];
    //tree->resetPositionAndTags();

	//T worst_dist = INFINITY;
	auto& worst_dist = worst_dist_[0];

	best_dist_pp->buffers[0] = buffered_dists;
	best_dist_pp->buffers[1] = buffered_dists + nr_nns_searches;
	best_knn_pp.buffers[0] = buffered_knn;
	best_knn_pp.buffers[1] = buffered_knn + nr_nns_searches;
	for(auto j = blockidx; j < nr_query; j += grid_size)
	{
		//Fetch vars from global memory
		const Vec<T, dims> point = points_query[j];
		*point_proj = points_query[j];

		if(tidx == 0)
			worst_dist = INFINITY;

		__syncthreads();
		//Buffer everything in shared memory and reset the tree
		fillKernel<T>(buffered_dists, buffered_dists + 2*nr_nns_searches, INFINITY);
		assert(nr_nodes > 0);
		fillKernel<NodeTag>(tags, tags + ((nr_nodes - 1)/4) + 1, NodeTag::uncharted);
		if(tidx == 0)
		{
			tree->current_lin_ind = tree->current_level = 0;
		}
		
		//Fetch the current leaf and descent
		__syncthreads();
		if(tidx == 0)
			findNextLeaf<T, T_calc, dims>(*tree, point, point_proj[0], worst_dist);
		__syncthreads();

		do
		{
			//printf("%d, %d\n", tree->getCurrentLevel(), tree->getTotalLevels() - 1);
			assert(tree->getCurrentLevel() == tree->getTotalLevels() - 1);
			assert(tree->isLeafParent());

			const auto current_partition = tree->getCurrentConstPartition();
			const auto current_axis = current_partition.axis_split;
			const bool lower_than_median = point[current_axis] < current_partition.median;
			const auto leaf = (lower_than_median ? tree->getLeftLeaf() : tree->getRightLeaf());

			//Now compute each leaf with all threads in the current block
			compQuadrDistLeafPartitionBlockwise<T, T_calc, dims>(point, leaf, local_dist_buf, *best_dist_pp, best_knn_pp,
					 nr_nns_searches, worst_dist);

			if(tidx == 0)
			{
				off_leaf_necessary[0] = partitionNecessary<T, T_calc, dims>(point, point_proj[0], current_partition, worst_dist);
			}

			__syncthreads();
			if(off_leaf_necessary[0])
			{
				//printf("Off partition necessary\n");
				const PartitionLeaf<T, dims>& leaf = (!lower_than_median ? tree->getLeftLeaf() : tree->getRightLeaf());
				//printf("Comp Dist 2\n");
				compQuadrDistLeafPartitionBlockwise<T, T_calc, dims>(point, leaf, local_dist_buf, *best_dist_pp, best_knn_pp,
					nr_nns_searches, worst_dist);
			}

			//__syncthreads();
			if(tidx == 0)
				findNextLeaf<T, T_calc, dims>(*tree, point, point_proj[0], worst_dist);

			__syncthreads();
		}while(tree->getCurrentLevel() != 0);
		
		//Copy back to global memory and proceed to next point
		point_i_knn_t* best_i = all_best_i_d + j * nr_nns_searches;
		T* best_dists = all_best_dists_d + j * nr_nns_searches;

		T* dist_slot = best_dist_pp->getCurrentSlot();
		point_i_t* knn_slot = best_knn_pp.getCurrentSlot();

		copyKernel(dist_slot, dist_slot + nr_nns_searches, best_dists);
		copyKernel(knn_slot, knn_slot + nr_nns_searches, best_i);
	}

	//if(tidx == 0)
	//	delete local_dist_buf_pointer[0];
}

template
<typename T, typename T_calc, dim_t dims>
void KDTreeKNNGPUSearch(PartitionInfoDevice<T, dims>* partition_info,
                    const point_i_t nr_query, 
                    const std::array<T, dims>* points_query, T * dist, point_i_t* idx, const point_i_knn_t nr_nns_searches)
{
	//TODO: Dynamic implementation
	/*if(partition_info->nr_partitions > max_partitions || partition_info->nr_leaves > max_leaves)
	{
		throw std::runtime_error("Error, please reduce number of levels...");
	}*/

	if(nr_nns_searches > max_nr_nns_searches)
		throw std::runtime_error("TODO: Maximum number of NNs searches currently restricted");

	//gpuErrchk(cudaMemcpyAsync(partition_info_copy, partition_info, sizeof(PartitionInfoDevice<T, dims>), cudaMemcpyDeviceToDevice));
	initArray<T><<<dim3(16, 16),dim3(32, 32)>>>(dist, INFINITY, nr_query*nr_nns_searches);

	#ifndef NDEBUG
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
	#endif

	dim3 grid_dims(32, 32);
	dim3 block_dims(8, 8);

	/*#ifdef PROFILE_KDTREE
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	#endif*/

	const auto points_query_eig = reinterpret_cast<const Vec<T, dims>*>(points_query);
	KDTreeKernel<T, T_calc, dims><<<grid_dims, block_dims>>>(partition_info, nr_query, points_query_eig, dist, idx, nr_nns_searches);

	#ifndef NDEBUG
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
	#endif

	/*#ifdef PROFILE_KDTREE
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf ("Time for the kernel: %f ms\n", time);
	#endif*/
}

template void compQuadrDistLeafPartition<float, float, 3>(const std::array<float, 3>& point, const PartitionLeaf<float, 3>& partition_leaf,
                                    float* best_dists, point_i_knn_t* best_idx,
                                    const point_i_knn_t nr_nns_searches);
                                    
template void compQuadrDistLeafPartition<double, double, 3>(const std::array<double, 3>& point, const PartitionLeaf<double, 3>& partition_leaf,
                                    double* best_dists, point_i_knn_t* best_idx,
                                    const point_i_knn_t nr_nns_searches);

template void KDTreeKNNGPUSearch<float, float, 1>(PartitionInfoDevice<float, 1>* partition_info,
                    const point_i_t nr_query, 
                    const std::array<float, 1>* points_query, float * dist, point_i_t* idx, const point_i_knn_t nr_nns_searches);
template void KDTreeKNNGPUSearch<double, double, 1>(PartitionInfoDevice<double, 1>* partition_info, 
    const point_i_t nr_query, 
    const std::array<double, 1>* points_query, double * dist, point_i_t* idx, const point_i_knn_t nr_nns_searches);
	
template void KDTreeKNNGPUSearch<float, float, 2>(PartitionInfoDevice<float, 2>* partition_info,
		const point_i_t nr_query, 
		const std::array<float, 2>* points_query, float * dist, point_i_t* idx, const point_i_knn_t nr_nns_searches);
template void KDTreeKNNGPUSearch<double, double, 2>(PartitionInfoDevice<double, 2>* partition_info,
	const point_i_t nr_query, 
	const std::array<double, 2>* points_query, double * dist, point_i_t* idx, const point_i_knn_t nr_nns_searches);

template void KDTreeKNNGPUSearch<float, float, 3>(PartitionInfoDevice<float, 3>* partition_info,
	const point_i_t nr_query, 
	const std::array<float, 3>* points_query, float * dist, point_i_t* idx, const point_i_knn_t nr_nns_searches);
template void KDTreeKNNGPUSearch<double, double, 3>(PartitionInfoDevice<double, 3>* partition_info,
	const point_i_t nr_query, 
	const std::array<double, 3>* points_query, double * dist, point_i_t* idx, const point_i_knn_t nr_nns_searches);

template PartitionInfoDevice<float, 1>* copyPartitionToGPU(const PartitionInfo<float, 1>& partition_info);
template PartitionInfoDevice<float, 2>* copyPartitionToGPU(const PartitionInfo<float, 2>& partition_info);
template PartitionInfoDevice<float, 3>* copyPartitionToGPU(const PartitionInfo<float, 3>& partition_info);
template PartitionInfoDevice<double, 1>* copyPartitionToGPU(const PartitionInfo<double, 1>& partition_info);
template PartitionInfoDevice<double, 2>* copyPartitionToGPU(const PartitionInfo<double, 2>& partition_info);
template PartitionInfoDevice<double, 3>* copyPartitionToGPU(const PartitionInfo<double, 3>& partition_info);

template std::tuple<float*, point_i_t*, float*> copyData<float, 3>(const std::vector<float>& result_dists, const std::vector<point_i_t>& result_idx, const std::vector<std::array<float, 3>>&);
template std::tuple<double*, point_i_t*, double*> copyData<double, 3>(const std::vector<double>& result_dists, const std::vector<point_i_t>& result_idx, const std::vector<std::array<double, 3>>&);

template void freePartitionFromGPU(PartitionInfoDevice<float, 1>* partition_info);
template void freePartitionFromGPU(PartitionInfoDevice<float, 2>* partition_info);
template void freePartitionFromGPU(PartitionInfoDevice<float, 3>* partition_info);
template void freePartitionFromGPU(PartitionInfoDevice<double, 1>* partition_info);
template void freePartitionFromGPU(PartitionInfoDevice<double, 2>* partition_info);
template void freePartitionFromGPU(PartitionInfoDevice<double, 3>* partition_info);

template void freeGPUArray(float* arr);
template void freeGPUArray(double* arr);
template void freeGPUArray(point_i_knn_t* arr);

