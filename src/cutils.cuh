#pragma once

/**
 * @file cutils.cuh
 * @author Thomas Grandits
 * @brief Cuda utility functions used throughout the program
 * @version 0.1
 * @date 2020-12-16
 * 
 * @copyright Copyright (c) 2020
 * 
 */

// includes CUDA Runtime
//#include <cuda_runtime.h>
#include "nndistance.hpp"
#include "kdtree.hpp"

//https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
/**
 * @brief Macro for CUDA assertions
 * 
 */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

/**
 * @brief Helper function to enable asserts in Debug mode for CUDA internal functions
 * 
 * @param code Error code
 * @param file file name
 * @param line line in the file
 * @param abort Flag to indicate if the program should be aborted if an error occurs
 */
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   //#ifndef NDEBUG
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
   //#endif
}

template <typename T>
CUDA_HOSTDEV inline void assertEqual(const T lhs, const T rhs)
{
	if (lhs != rhs)
	{
		printf("%f != %f\n", lhs, rhs);
		assert(false);
	}
}

//https://stackoverflow.com/questions/10589925/initialize-device-array-in-cuda
/**
 * @brief Initializes an array with a given value
 * 
 * @tparam T Type of the array and fill values
 * @param d_ptr Pointer to the CUDA array
 * @param fill_val Value to fill the array with
 * @param nr_elems Nr-elements to fill
 */
template <typename T>
__global__ void initArray(T* d_ptr, const T fill_val, const size_t nr_elems)
{
	int tidx = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;

	for(; tidx < nr_elems; tidx += stride)
	{
		d_ptr[tidx] = fill_val;
	}
}


template 
<typename T>
inline void allocGPUMemory(T** dev_gpu_buffer, const size_t nr_elems)
{
  gpuErrchk(cudaMalloc(dev_gpu_buffer, nr_elems*sizeof(T)));
}

/**
 * @brief Allocates an array on the target device and copies the source array to that position
 * 
 * @tparam T Type of the array
 * @tparam transfer_t Data transfer source and location (see cudaMemcpyKind)
 * @param array_s Source array
 * @param nr_elems Source array length
 * @return T* Pointer to the newly allocated array, either on the host or device
 */
template <typename T, cudaMemcpyKind transfer_t>
inline T* copyArrayBetweenGPUCPU(const T* array_s, const size_t nr_elems)
{
    //static_assert(transfer_t == cudaMemcpyHostToDevice || transfer_t == cudaMemcpyDeviceToHost, "Not yet implemented");
    T* array_d;
    const auto array_size = nr_elems * sizeof(T);

    if /*constexpr*/(transfer_t == cudaMemcpyHostToDevice || transfer_t == cudaMemcpyDeviceToDevice)
      allocGPUMemory(&array_d, array_size);
    else if /*constexpr*/(transfer_t == cudaMemcpyDeviceToHost  || transfer_t == cudaMemcpyHostToHost)
      array_d = reinterpret_cast<T*>(malloc(array_size)); //array_d = new T[nr_elems];

    gpuErrchk(cudaMemcpy(array_d, array_s, array_size, transfer_t));
    return array_d;
}

/**
 * @brief Copies an array from host to device
 * 
 * @tparam T Array type
 * @param array_h Host array
 * @param nr_elems Array size
 * @return T* Pointer to the new array
 */
template <typename T>
inline T* copyArrayToDevice(const T* array_h, const size_t nr_elems)
{
    return copyArrayBetweenGPUCPU<T, cudaMemcpyHostToDevice>(array_h, nr_elems);
}

/**
* @brief Copies an array from device to host
* 
* @tparam T Array type
* @param array_d Device array
* @param nr_elems Array size
* @return T* Pointer to the new array
*/
template <typename T>
inline T* copyArrayToHost(const T* array_d, const size_t nr_elems)
{
    return copyArrayBetweenGPUCPU<T, cudaMemcpyDeviceToHost>(array_d, nr_elems);
}

/**
* @brief Frees the given device array
* 
* @tparam T Array type
* @param dev_gpu_buffer Device array
*/
template 
<typename T>
void freeGPUMemory(T* dev_gpu_buffer)
{
	gpuErrchk(cudaFree(dev_gpu_buffer));
}

/**
 * @brief Copies an array from src to dest in parallel
 * 
 * @tparam T Array type
 * @param src Source array
 * @param src_end End of the source array (exclusive)
 * @param dest Destination array
 */
template <typename T>
__device__ inline void copyKernel(const T* src, const T* src_end, T* dest)
{
	//2D indices
	//const int blockidx = blockIdx.x + blockIdx.y*gridDim.x;
	const int block_size = blockDim.x * blockDim.y; // * blockDim.z;
	const int tidx = threadIdx.y * blockDim.x + threadIdx.x;

	const auto dist = src_end - src;
  assert(dist >= 0);
  assert(!(src <= dest && dest <= src_end)); //Non overlapping arrays

	for(size_t i = tidx; i < dist; i += block_size)
	{
		dest[i] = src[i];
	}
}

/**
 * @brief Initializes an array with a given value
 * 
 * @tparam T Type of the array and fill values
 * @param src Pointer to the CUDA array
 * @param src_end End of the source array (exclusive)
 * @param val Value to fill
 */
template <typename T>
__device__ inline void fillKernel(T* src, T* src_end, T val)
{
	//2D indices
	//const int blockidx = blockIdx.x + blockIdx.y*gridDim.x;
	const int block_size = blockDim.x * blockDim.y; // * blockDim.z;
	const int tidx = threadIdx.y * blockDim.x + threadIdx.x;

	const auto dist = src_end - src;
	assert(dist >= 0);

	for(size_t i = tidx; i < dist; i += block_size)
	{
		src[i] = val;
	}
}

/**
 * @brief Parallely inserts an element into an array, shifting all entries from that position onwards to the right.
 *        This is performed using two arrays, which are required to be non-overlapping. The last element will be discarded.
 * 
 * @tparam T Type of the array
 * @param source_arr Source array where the new element will be inserted
 * @param target_arr Target array that has enough space to hold the source array. Must be non-overlapping with source_arr!
 * @param array_length Length of both source and target array
 * @param insert_val Value to insert
 * @param insertion_idx Index where to insert
 */
template <typename T>
__device__ void insertAndShiftArrayRight(const T* source_arr, T* target_arr, const point_i_knn_t array_length, const T insert_val, 
                                          const point_i_knn_t insertion_idx)
{
	const int block_size = blockDim.x * blockDim.y; // * blockDim.z;
	const int tidx = threadIdx.y * blockDim.x + threadIdx.x;

  //TODO: Reintroduce assertion
  //assert(!(source_arr <= target_arr && target_arr <= source_arr + array_length)); //Non overlapping arrays
  assert(insertion_idx < array_length);

	for(point_i_t arr_i = tidx; arr_i < array_length; arr_i+=block_size)
  {
    const auto target_i = arr_i;
    const auto src_i = arr_i - (arr_i > insertion_idx ? 1 : 0);
    target_arr[target_i] = (arr_i != insertion_idx ? source_arr[src_i] : insert_val);
	}
}

/**
 * @brief Similar to \ref insertAndShiftArrayRight(const T*, T*, const point_i_knn_t, const T,  const point_i_knn_t) "insertAndShiftArrayRight",
 *        but the new last element will be saved to new_last_elem
 * 
 * @tparam T 
 * @tparam sync_before_write Specifies if __syncthreads is to be called before overwriting new_last_elem
 * @param source_arr 
 * @param target_arr 
 * @param array_length 
 * @param insert_val 
 * @param insertion_idx 
 * @param new_last_elem New righmost element of the array, after the shifting operation has been performed.
 */
template <typename T, bool sync_before_write = false>
__device__ void insertAndShiftArrayRight(const T* source_arr, T* target_arr, const point_i_knn_t array_length, const T insert_val, 
                                          const point_i_knn_t insertion_idx, T& new_last_elem)
{
	const int block_size = blockDim.x * blockDim.y; // * blockDim.z;
	const int tidx = threadIdx.y * blockDim.x + threadIdx.x;

  //TODO: Reintroduce assertion
  //assert(!(source_arr <= target_arr && target_arr <= source_arr + array_length)); //Non overlapping arrays
  assert(insertion_idx < array_length);

  //TODO: For more efficiency, this could be moved inside the loop. Only meant to protect from race conditions on new_last_elem
  if (sync_before_write)
	  __syncthreads();

  for(point_i_t arr_i = tidx; arr_i < array_length; arr_i+=block_size)
  {
    const auto target_i = arr_i;
    const auto src_i = arr_i - (arr_i > insertion_idx ? 1 : 0);
    const T copied_val = (arr_i != insertion_idx ? source_arr[src_i] : insert_val);
    target_arr[target_i] = copied_val;

	if (target_i == array_length - 1)
	{
		new_last_elem = copied_val;
	}
  }
}

/**
 * @brief Parallely computes the squared distances between a single point and multiple other points
 * 
 * @tparam T Type of the points
 * @tparam dims Dimensions of the points
 * @param point The point for which all ref_leaf distances will be computed
 * @param ref_leaf Points for which all point distances will be computed
 * @param nr_ref Number of points in ref_leaf
 * @param dest Destination array for the computed squared distances
 */
template <typename T, uint32_t dims>
__device__ inline void compDists(const Vec<T, dims>& point, const Vec<T, dims>* ref_leaf, const size_t nr_ref, T* dest)
{
	//2D indices
	//const int blockidx = blockIdx.x + blockIdx.y*gridDim.x;
	const int block_size = blockDim.x * blockDim.y; // * blockDim.z;
	const int tidx = threadIdx.y * blockDim.x + threadIdx.x;

	for(size_t i = tidx; i < nr_ref; i += block_size)
	{
		dest[i] = (point - ref_leaf[i]).squaredNorm();
	}
}

/**
 * @brief Updates the current ordered KNN list, inserting a new point if its distance is smaller than any of the present distances
 * 
 * @tparam T Type of the distances
 * @tparam nr_nns_searches Number of nearest neighbors to search for (=k)
 * @tparam check_before_insertion Flag if the new distance should be first compared to the worst distance before insertion 
 *                                (setting this to false can potentially save some time if you know the value will surely be placed in the list).
 * @param d Distance of the new point
 * @param k Index of the new point
 * @param best_dists Current best distances for the KNN-search
 * @param best_i Current best indices for the KNN-search
 * @return uint32_t The index where the new point was inserted
 */
template 
<typename T, uint32_t nr_nns_searches, bool check_before_insertion=true>
__device__ uint32_t updateKNNLists(const T d, const int k, T* best_dists, uint32_t* best_i)
{
  const auto insertion_idx = knnInsertionStatic<T, nr_nns_searches, check_before_insertion>(d, best_dists);
			
  if(!check_before_insertion || insertion_idx < nr_nns_searches)
  {
    moveBackward(best_dists + insertion_idx, best_dists + nr_nns_searches - 1, best_dists + nr_nns_searches);
    moveBackward(best_i + insertion_idx, best_i + nr_nns_searches - 1, best_i + nr_nns_searches);
    best_dists[insertion_idx] = d;
    best_i[insertion_idx] = k;
  }
  
  return insertion_idx;
}


/**
 * @brief Similar to \ref updateKNNLists 'updateKNNLists', but accepts a dynamic amount of nr_nns_searches (=k)
 * 
 * @tparam T Type of the distances
 * @tparam check_before_insertion Flag if the new distance should be first compared to the worst distance before insertion 
 *                                (setting this to false can potentially save some time if you know the value will surely be placed in the list).
 * @param d Distance of the new point
 * @param k Index of the new point
 * @param best_dists Current best distances for the KNN-search
 * @param best_i Current best indices for the KNN-search
 * @param nr_nns_searches Number of nearest neighbors to search for (=k)
 * @return uint32_t The index where the new point was inserted
 */
template 
<typename T, bool check_before_insertion=true>
__device__ uint32_t updateKNNListsDynamic(const T d, const int k, T* best_dists, uint32_t* best_i, const uint32_t nr_nns_searches)
{
  const auto insertion_idx = knnInsertionDynamic<T, check_before_insertion>(d, best_dists, nr_nns_searches);
			
  if(!check_before_insertion || insertion_idx < nr_nns_searches)
  {
    moveBackward(best_dists + insertion_idx, best_dists + nr_nns_searches - 1, best_dists + nr_nns_searches);
    moveBackward(best_i + insertion_idx, best_i + nr_nns_searches - 1, best_i + nr_nns_searches);
    best_dists[insertion_idx] = d;
    best_i[insertion_idx] = k;
  }
  
  return insertion_idx;
}
