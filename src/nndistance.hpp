#pragma once
/**
 * @file nndistance.hpp
 * @author Thomas Grandits (thomas.grandits@icg.tugraz.at)
 * @brief Helper KNN functions.
 * @version 0.1
 * @date 2020-12-16
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#include <cstdint>

//https://stackoverflow.com/questions/32014839/how-to-use-a-cuda-class-header-file-in-both-cpp-and-cuda-modules
#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

template <typename T>
struct DevKnnMem
{
	T* best_dists;
	uint32_t* best_i;
};

//https://en.cppreference.com/w/cpp/algorithm/upper_bound
/**
 * @brief Similar to https://en.cppreference.com/w/cpp/algorithm/upper_bound , but also works on 
 *        the CUDA device. Assumes the range to be sorted, but has O(log n) runtime in return.
 * 
 * @tparam ForwardIt Type of the forward iterator
 * @tparam T Type of the data
 * @param first Beginning of the range where to find the upper bound
 * @param last End (exlusive) of the range where to find the upper bound
 * @param value The value for which we want to find the upper bound
 * @return ForwardIt The upper bound location. =End if outside
 */
template<class ForwardIt, class T> 
CUDA_HOSTDEV ForwardIt upperBound(ForwardIt first, ForwardIt last, const T& value)
{
    ForwardIt it;
    uint32_t count, step;
    count = last - first;
 
    while (count > 0) {
        it = first; 
        step = count / 2; 
        it += step;
        if (value >= *it) {
            first = ++it;
            count -= step + 1;
        } 
        else
            count = step;
    }
    return first;
}


/**
 * @brief Inserts the new distance dist in the ordered KNN distances best_dists and returns the insertion index.
 *        If dist > best_dists[K-1], the insertion idx will be K. The search uses upperBound to achieve logarithmic
 *        insertion time.
 * 
 * @tparam check_before_insertion If true, the function will check dist > best_dists[K-1] before beginning the computation.
 * @param dist Distance to insert
 * @param best_dists Array of size K, with the current closest KNN
 * @param nr_nns_searches Number of nearest neighbors to search for (=K)
 * @return uint32_t Insertion index
 */
template
<typename T, bool check_before_insertion=true>
CUDA_HOSTDEV uint32_t knnInsertionDynamic(const T dist, T* best_dists, const uint32_t nr_nns_searches)
{
  if(check_before_insertion && dist >= best_dists[nr_nns_searches - 1])
    return nr_nns_searches;
  else
  {
    const auto insertion_it = upperBound(best_dists, best_dists + nr_nns_searches, dist);
    return insertion_it - best_dists;
  }
}


/**
 * @brief Similar to \ref knnInsertionDynamic 'knnInsertionDynamic', but for a static number of nearest
 *        neighbors (=K).
 */
template
<typename T, uint32_t nr_nns_searches, bool check_before_insertion=true>
CUDA_HOSTDEV uint32_t knnInsertionStatic(const T dist, T* best_dists)
{
  if(check_before_insertion && dist >= best_dists[nr_nns_searches - 1])
    return nr_nns_searches;
  else
  {
    const auto insertion_it = upperBound<T*, T>(best_dists, best_dists + nr_nns_searches, dist);
    return insertion_it - best_dists;    
  }
}

/**
 * @brief Moves the elements from the array marked by begin and end_exlusive to target. This is done backward,
 *        to allow for right shifts of the array.
 * 
 * @tparam T Type of the array
 * @param begin Beginning of the array, inclusive
 * @param end_exclusive End of the array, exclusive
 * @param target End of the target array, exlusive
 * @return Last element that was overwritten (i.e. target at the end of the algorithm)
 */
template
<typename T>
inline CUDA_HOSTDEV T* moveBackward(const T* begin, T* end_exclusive, T* target)
{
	T* current = end_exclusive;

	while(current > begin)
	{
		*(--target) = std::move(*(--current));
	}

	return target;
}
