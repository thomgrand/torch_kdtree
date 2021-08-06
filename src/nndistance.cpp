#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>
#include "nndistance.hpp"

REGISTER_OP("KnnDistance")
  .Attr("T: realnumbertype = DT_FLOAT")
	.Input("points_ref: T")
	.Input("points_query: T")
	.Attr("nr_nns_searches: int = 1")
	.Output("dist: T")
	.Output("idx: int32")
		.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c)
		{
			int nr_nns_searches; // = std::stoi(nr_nns_searches_str);
			c->GetAttr("nr_nns_searches", &nr_nns_searches);
			c->set_output(0, c->Matrix(c->Dim(c->input(1), 0), nr_nns_searches));
			c->set_output(1, c->Matrix(c->Dim(c->input(1), 0), nr_nns_searches));
      		return tensorflow::Status::OK();
		});
using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;


template
<typename T, typename T_calc>
static void nnsearchDynamic(const int n, const int m, const T * points_ref, const T * points_query, T * dist, int* idx,
                            const uint32_t nr_nns_searches){
	#pragma omp parallel for
	for (uint32_t j=0;j<m;j++){
		T_calc x1=points_query[(j)*3+0];
		T_calc y1=points_query[(j)*3+1];
		T_calc z1=points_query[(j)*3+2];
		std::vector<T_calc> best_dists(nr_nns_searches);
		std::vector<uint32_t> best_i(nr_nns_searches);
    
		std::fill(best_dists.begin(), best_dists.end(), std::numeric_limits<T_calc>::infinity());    

		for (uint32_t k=0;k<n;k++){
			T_calc x2=points_ref[(k)*3+0]-x1;
			T_calc y2=points_ref[(k)*3+1]-y1;
			T_calc z2=points_ref[(k)*3+2]-z1;
			T_calc d=x2*x2+y2*y2+z2*z2;
			const auto insertion_idx = knnInsertionDynamic<T_calc>(d, best_dists.data(), nr_nns_searches);
			
			if(insertion_idx < nr_nns_searches)
			{
				std::move_backward(best_dists.begin() + insertion_idx, best_dists.end() - 1, best_dists.end());
				std::move_backward(best_i.begin() + insertion_idx, best_i.end() - 1, best_i.end());
				best_dists[insertion_idx] = d;
				best_i[insertion_idx] = k;
			}
		}
		std::copy(best_dists.begin(), best_dists.end(), dist + j*nr_nns_searches);
		std::copy(best_i.begin(), best_i.end(), idx + j*nr_nns_searches); //TODO: Type mismatch
	}
}


template <typename T>
class KnnDistanceOp : public OpKernel{
	public:
		explicit KnnDistanceOp(OpKernelConstruction* context):OpKernel(context)
		{
      		OP_REQUIRES_OK(context, context->GetAttr("nr_nns_searches", &nr_nns_searches));
		}
		
		void checkParamsAndReturnVars(OpKernelContext * context){
      
			const Tensor& points_ref_tensor=context->input(0);
			const Tensor& points_query_tensor=context->input(1);

			OP_REQUIRES(context,points_ref_tensor.dims()==2,errors::InvalidArgument("NnDistance requires xyz1 be of shape (#points,3)"));
			OP_REQUIRES(context,points_ref_tensor.shape().dim_size(1)==3,errors::InvalidArgument("NnDistance only accepts 3d point set xyz1"));
			int n=points_ref_tensor.shape().dim_size(0);
			OP_REQUIRES(context,points_query_tensor.dims()==2,errors::InvalidArgument("NnDistance requires xyz2 be of shape (#points,3)"));
			OP_REQUIRES(context,points_query_tensor.shape().dim_size(1)==3,errors::InvalidArgument("NnDistance only accepts 3d point set xyz2"));
			int m=points_query_tensor.shape().dim_size(0);
			auto points_ref_flat=points_ref_tensor.flat<T>();
			const T * xyz1=&points_ref_flat(0);
			auto points_query_flat=points_query_tensor.flat<T>();
			const T * points_query=&points_query_flat(0);
			Tensor * dist_tensor=NULL;
			Tensor * idx_tensor=NULL;
			OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{m, nr_nns_searches},&dist_tensor));
			OP_REQUIRES_OK(context,context->allocate_output(1,TensorShape{m, nr_nns_searches},&idx_tensor));
			auto dist_flat=dist_tensor->flat<T>();
			auto idx_flat=idx_tensor->flat<int>();
			T * dist=&(dist_flat(0));
			int * idx=&(idx_flat(0));
			
			vars = std::make_tuple(n, m, xyz1, points_query, dist, idx);
		}
		void Compute(OpKernelContext * context)override{
			
			
			/*auto vars =*/ this->checkParamsAndReturnVars(context);
			auto n = std::get<0>(vars);
			auto m = std::get<1>(vars);
			auto points_ref = std::get<2>(vars);
			auto points_query = std::get<3>(vars);
			auto dist = std::get<4>(vars);
			auto idx = std::get<5>(vars);
			
			nnsearchDynamic<T, T>(n,m,points_ref,points_query,dist,idx, nr_nns_searches);
		}

	protected:
		//int nr_nns_searches;
		std::tuple<int, int, const T*, const T*, T*, int*>  vars;
		int nr_nns_searches = 20;
};

REGISTER_KERNEL_BUILDER(Name("KnnDistance").Device(DEVICE_CPU).TypeConstraint<float>("T"), KnnDistanceOp<float>); 
REGISTER_KERNEL_BUILDER(Name("KnnDistance").Device(DEVICE_CPU).TypeConstraint<double>("T"), KnnDistanceOp<double>);

template
<typename T, typename T_calc>
void NmDistanceKernelLauncherDynamic(int n, int m,const T * xyz, const T * xyz2, T * result, int * result_i, const uint32_t nr_nns_searches,
										DevKnnMem<T_calc>& dev_knn_mem);


template 
<typename T>
void allocNNDistanceGPUMemory(T** dev_gpu_buffer, const uint32_t nr_nns_searches);

template 
<typename T>
void freeGPUMemory(T* dev_gpu_buffer);

template <typename T>
class KnnDistanceGpuOp : public KnnDistanceOp<T>{ //OpKernel{
	public:
		explicit KnnDistanceGpuOp(OpKernelConstruction* context):KnnDistanceOp<T>(context)
		{
			//std::cout << this->dev_knn_gpu_buffer.best_dists << ", " << this->dev_knn_gpu_buffer.best_i << std::endl;
			allocNNDistanceGPUMemory(&(this->dev_knn_gpu_buffer.best_dists), this->nr_nns_searches);
			allocNNDistanceGPUMemory(&(this->dev_knn_gpu_buffer.best_i), this->nr_nns_searches);
			//std::cout << this->dev_knn_gpu_buffer.best_dists << ", " << this->dev_knn_gpu_buffer.best_i << std::endl;
		}
		void Compute(OpKernelContext * context)override{
			
			/*auto vars =*/ this->checkParamsAndReturnVars(context);
			auto n = std::get<0>(this->vars);
			auto m = std::get<1>(this->vars);
			auto points_ref = std::get<2>(this->vars);
			auto points_query = std::get<3>(this->vars);
			auto dist = std::get<4>(this->vars);
			auto idx = std::get<5>(this->vars);
			
			NmDistanceKernelLauncherDynamic<T, T>(n, m, points_ref, points_query, dist, idx, this->nr_nns_searches, this->dev_knn_gpu_buffer);
		}

		~KnnDistanceGpuOp()
		{
			freeGPUMemory(this->dev_knn_gpu_buffer.best_dists);
			freeGPUMemory(this->dev_knn_gpu_buffer.best_i);
		}

	protected:
		DevKnnMem<T> dev_knn_gpu_buffer;
};

REGISTER_KERNEL_BUILDER(Name("KnnDistance").Device(DEVICE_GPU).TypeConstraint<float>("T"), KnnDistanceGpuOp<float>); 
REGISTER_KERNEL_BUILDER(Name("KnnDistance").Device(DEVICE_GPU).TypeConstraint<double>("T"), KnnDistanceGpuOp<double>);
