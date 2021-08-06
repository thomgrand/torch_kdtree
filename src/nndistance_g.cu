#define EIGEN_USE_GPU
#include <unsupported/Eigen/CXX11/Tensor>
#include "nndistance.hpp"
#include "cutils.cuh"

template 
<typename T, typename T_calc, uint32_t nr_nns_searches>
__global__ void NmDistanceKernel(int n, const T* points_ref, int m, const T* points_query, T* result, int* result_i){
	const int batch=512*sizeof(T)/4;
	assert(nr_nns_searches <= n);
	const auto point_dims = 3;
	__shared__ T ref_buf[batch*point_dims];

	//2D indices
	const int blockidx = blockIdx.x + blockIdx.y*gridDim.x;
	const int block_size = blockDim.x * blockDim.y; // * blockDim.z;
	const int tidx = threadIdx.y * blockDim.x + threadIdx.x;
	uint32_t best_i[nr_nns_searches]; //int best_i=0;
	T_calc best_dists[nr_nns_searches]; //float best=0;
	//print("")
  
	for(int j = tidx; j < m; j+=block_size) //blockDim.y) //blockDim.x*gridDim.y) //for (int j=threadIdx.x+blockIdx.y*blockDim.x;
	{
		T_calc x1=points_query[j*point_dims+0];
		T_calc y1=points_query[j*point_dims+1];
		T_calc z1=points_query[j*point_dims+2];
		T_calc slowest_time = INFINITY;

		#pragma unroll
		for(int current_nn = 0; current_nn < nr_nns_searches; current_nn++) { best_dists[current_nn] = INFINITY; }

		for (int k2=0;k2<n;k2+=batch){
			int end_k = min(n, k2 + batch) - k2;
			//printf("end_k %d\n", end_k);
			//printf("Thread Idx: %d\n", tidx);
			__syncthreads(); 
			for(int ref_i = tidx; ref_i < end_k; ref_i += min(block_size, m)){ 
				#pragma unroll
				for(int dim_i = 0; dim_i < point_dims; dim_i++) {ref_buf[point_dims*ref_i + dim_i] = points_ref[(k2 + ref_i) * point_dims + dim_i];}
			}
			__syncthreads(); 
	  
			const auto inner_stride = 4;
			for (int k = 0; k < end_k; k += inner_stride)
			{
			  #pragma unroll
			  for(int l=0; l<inner_stride; l++)
			  {
				  if(k+l >= end_k)
            		break;
					  
				  T_calc x2 = ref_buf[(k + l) * point_dims + 0] - x1;
				  T_calc y2 = ref_buf[(k + l) * point_dims + 1] - y1;
				  T_calc z2 = ref_buf[(k + l) * point_dims + 2] - z1;
				  T_calc d = x2*x2+y2*y2+z2*z2;
				  
				if(d < slowest_time)
				{
					updateKNNLists<T_calc, nr_nns_searches, false>(d, k+k2+l, best_dists, best_i);
					slowest_time = best_dists[nr_nns_searches-1];
				}
			  }
		  }
		}
		#pragma unroll
		for(int i = 0; i < nr_nns_searches; i++) {result[j*nr_nns_searches + i] = best_dists[i];}
    
		#pragma unroll
		for(int i = 0; i < nr_nns_searches; i++) {result_i[j*nr_nns_searches + i] = best_i[i];}
		__syncthreads();
	}
}

template 
<typename T, typename T_calc>
__global__ void NmDistanceKernelDynamic(int n, const T* points_ref, int m, const T* points_query, T* result, int* result_i, const uint32_t nr_nns_searches,
	T_calc* dev_all_best_dists, uint32_t* dev_all_best_i){
	const int batch=512*sizeof(T)/4;
	assert(nr_nns_searches <= n);
	const auto point_dims = 3;
	__shared__ T ref_buf[batch*point_dims];

	//2D indices
	const int blockidx = blockIdx.x + blockIdx.y*gridDim.x;
	const int block_size = blockDim.x * blockDim.y; 
	const int tidx = threadIdx.y * blockDim.x + threadIdx.x;
	uint32_t* best_i = dev_all_best_i + tidx * nr_nns_searches;
	T_calc* best_dists = dev_all_best_dists + tidx * nr_nns_searches;
  
	for(int j = tidx; j < m; j+=block_size)
	{
		T_calc x1=points_query[j*point_dims+0];
		T_calc y1=points_query[j*point_dims+1];
		T_calc z1=points_query[j*point_dims+2];
		T_calc slowest_time = INFINITY;

		for(int current_nn = 0; current_nn < nr_nns_searches; current_nn++) { best_dists[current_nn] = INFINITY;}

		for (int k2=0;k2<n;k2+=batch){
			int end_k = min(n, k2 + batch) - k2;

			__syncthreads(); 
			for(int ref_i = tidx; ref_i < end_k; ref_i += min(block_size, m)){

				#pragma unroll
				for(int dim_i = 0; dim_i < point_dims; dim_i++) {ref_buf[point_dims*ref_i + dim_i] = points_ref[(k2 + ref_i) * point_dims + dim_i];}
			}
			__syncthreads(); 
	  
			const auto inner_stride = 4;
			for (int k = 0; k < end_k; k += inner_stride)
			{
			  #pragma unroll
			  for(int l=0; l<inner_stride; l++)
			  {
				  if(k+l >= end_k)
					break;
					  
				  //const auto offset = point_dims*l;
				  T_calc x2 = ref_buf[(k + l) * point_dims + 0] - x1;
				  T_calc y2 = ref_buf[(k + l) * point_dims + 1] - y1;
				  T_calc z2 = ref_buf[(k + l) * point_dims + 2] - z1;
				  T_calc d = x2*x2+y2*y2+z2*z2;
				  
				if(d < slowest_time)
				{
					updateKNNListsDynamic<T_calc, false>(d, k+k2+l, best_dists, best_i, nr_nns_searches);
					slowest_time = best_dists[nr_nns_searches-1];
				}
			  }
		  }
		}

		for(int i = 0; i < nr_nns_searches; i++) {result[j*nr_nns_searches + i] = best_dists[i];}
    

		for(int i = 0; i < nr_nns_searches; i++) {result_i[j*nr_nns_searches + i] = best_i[i];} 
		__syncthreads();
	}
}

template 
<typename T, typename T_calc, uint32_t nr_nns_searches>
void NmDistanceKernelLauncher(int n, int m,const T * points_ref, const T * points_query, T * result, int * result_i)
{
	//gpuErrchk(cudaMemcpy(d_array[i], h_array[i], array_size * sizeof(int), cudaMemcpyHostToDevice));
	//gpuErrchk( cudaMemset(result, static_cast<float>(INFINITY), m*nr_nns_searches*sizeof(float))); //Does not work (byte by byte)
	initArray<T><<<dim3(16, 16),dim3(32, 32)>>>(result, INFINITY, m*nr_nns_searches);
	//gpuErrchk( cudaPeekAtLastError() );
	//gpuErrchk( cudaDeviceSynchronize() );
	dim3 grid_dims(1, 1); //Has to stay one
	dim3 block_dims(32, 32);
	/*cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);*/

	NmDistanceKernel<T, T_calc, nr_nns_searches><<<grid_dims, block_dims,512>>>(n,points_ref,m,points_query,result,result_i);

	/*cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf ("Time for the kernel: %f ms\n", time);*/
	
	//gpuErrchk( cudaPeekAtLastError() );
	//gpuErrchk( cudaDeviceSynchronize() );
	//NmDistanceKernel<<<dim3(32,16,1),512>>>(m,xyz2,n,xyz,result2,result2_i); //Reverse search
}

const uint32_t knn_block_x = 32;
const uint32_t knn_block_y = 32;

template 
<typename T>
void allocNNDistanceGPUMemory(T** dev_gpu_buffer, const uint32_t nr_nns_searches)
{
	gpuErrchk(cudaMalloc(reinterpret_cast<void**>(dev_gpu_buffer), nr_nns_searches*sizeof(T)*knn_block_x*knn_block_y));
}

template void allocNNDistanceGPUMemory(float**, const uint32_t);
template void allocNNDistanceGPUMemory(double**, const uint32_t);
template void allocNNDistanceGPUMemory(uint32_t**, const uint32_t);
template void allocNNDistanceGPUMemory(uint64_t**, const uint32_t);

template void freeGPUMemory(float* dev_gpu_buffer);
template void freeGPUMemory(double* dev_gpu_buffer);
template void freeGPUMemory(uint32_t* dev_gpu_buffer);
template void freeGPUMemory(uint64_t* dev_gpu_buffer);

template 
<typename T, typename T_calc>
void NmDistanceKernelLauncherDynamic(int n, int m,const T * points_ref, const T * points_query, T * result, int * result_i, const uint32_t nr_nns_searches,
						DevKnnMem<T_calc>& dev_knn_mem)
{
	//gpuErrchk(cudaMemcpy(d_array[i], h_array[i], array_size * sizeof(int), cudaMemcpyHostToDevice));
	//gpuErrchk( cudaMemset(result, static_cast<float>(INFINITY), m*nr_nns_searches*sizeof(float))); //Does not work (byte by byte)
	initArray<T><<<dim3(16, 16),dim3(32, 32)>>>(result, INFINITY, m*nr_nns_searches);
	//gpuErrchk( cudaPeekAtLastError() );
	//gpuErrchk( cudaDeviceSynchronize() );
	dim3 grid_dims(1, 1); //Has to stay one
	dim3 block_dims(knn_block_x, knn_block_y);
	/*cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);*/

	NmDistanceKernelDynamic<T, T_calc><<<grid_dims, block_dims,512>>>(n,points_ref,m,points_query,result,result_i, nr_nns_searches,
		dev_knn_mem.best_dists, dev_knn_mem.best_i);

	/*cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf ("Time for the kernel: %f ms\n", time);
	
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );*/
	//NmDistanceKernel<<<dim3(32,16,1),512>>>(m,xyz2,n,xyz,result2,result2_i); //Reverse search
}


template void NmDistanceKernelLauncherDynamic<float, float>(int n, int m,const float * xyz, const float * xyz2, float * result,int * result_i,
                                                                const uint32_t nr_nns_searches, DevKnnMem<float>&);
template void NmDistanceKernelLauncherDynamic<double, double>(int n, int m,const double * xyz, const double * xyz2, double * result,int * result_i,
                                                                const uint32_t nr_nns_searches, DevKnnMem<double>&);
