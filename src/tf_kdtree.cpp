#include <cstdint>
#include <stdint.h>
#include "kdtree.hpp"
#include "kdtree_g.hpp"
#include "nndistance.hpp"

/*
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

REGISTER_OP("BuildKdTree")
    .Attr("T: realnumbertype = DT_FLOAT")
	.Input("points_ref: T")
	.Attr("levels: int = 5")
	.Output("points_kdtree: T")
	.Output("metadata_address_kdtree: uint32")
	.Output("shuffled_inds: uint32")
		.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c)
		{
			c->set_output(0, c->Matrix(c->Dim(c->input(0), 0), c->Dim(c->input(0), 1)));
			c->set_output(1, c->Matrix(1, 1));
			c->set_output(2, c->Matrix(c->Dim(c->input(0), 0), 1));
      		return tensorflow::Status::OK();
		});
	
REGISTER_OP("KdTreeKnnSearch")
    .Attr("T: realnumbertype = DT_FLOAT")
	.Input("points_query: T")
	.Attr("metadata_address_kdtree: int")
	.Attr("nr_nns_searches: int = 1")
	.Output("dist: T")
	.Output("idx: uint32")
		.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c)
		{
			int nr_nns_searches; 
			c->GetAttr("nr_nns_searches", &nr_nns_searches);
			c->set_output(0, c->Matrix(c->Dim(c->input(0), 0), nr_nns_searches));
			c->set_output(1, c->Matrix(c->Dim(c->input(0), 0), nr_nns_searches));
      		return tensorflow::Status::OK();
		});

using namespace tensorflow;
*/

template <typename T>
class PartitionShareholder
{
public:
    ~PartitionShareholder()
    {
        size_t part_i = 0;
        for(part_i = 0; part_i < partition_infos.size(); part_i++)
        {
            const auto dims = dimensions[part_i];
            const auto main_partition = partition_infos[part_i];

            if(dims == 1)
                delete reinterpret_cast<PartitionInfo<T, 1>*>(main_partition);
            else if(dims == 2)
                delete reinterpret_cast<PartitionInfo<T, 2>*>(main_partition);
            else if(dims == 3)
                delete reinterpret_cast<PartitionInfo<T, 3>*>(main_partition);
        }
    }

    size_t addPartition(void* new_main_partition, const dim_t main_partition_dims, const int levels)
    {
        //std::cout << "Adding new partition... current size: " << partition_infos.size() << std::endl;
        partition_infos.push_back(new_main_partition);
        dimensions.push_back(main_partition_dims);
        partition_levels.push_back(levels);

        return partition_infos.size() - 1;
    }

    template <typename Ret = void*>
    Ret getPartition(const size_t part_nr)
    { 
        if(part_nr > partition_infos.size())
            throw std::runtime_error("You requested a non-present partition... This probably happened because you used a different precision for the KDtree, multiple GPUs or machines, which this library does not currently support");

        return reinterpret_cast<Ret>(partition_infos[part_nr]);
    }

    dim_t getPartitionDims(const size_t part_nr) const
    { 
		if(part_nr > partition_infos.size())
            throw std::runtime_error("You requested a non-present partition... This probably happened because you used a different precision for the KDtree, multiple GPUs or machines, which this library does not currently support");

		return dimensions[part_nr];  
	}

    int getPartitionLevels(const size_t part_nr) const
    { 
		if(part_nr > partition_infos.size())
            throw std::runtime_error("You requested a non-present partition... This probably happened because you used a different precision for the KDtree, multiple GPUs or machines, which this library does not currently support");

		return partition_levels[part_nr];  
	}

    size_t nrPartitions() const { return partition_infos.size();}

protected:
    std::vector<void*> partition_infos;
    std::vector<dim_t> dimensions;
    std::vector<int> partition_levels;
};

static PartitionShareholder<float> partition_share_float;
static PartitionShareholder<double> partition_share_double;

template <typename T>
class BuildKDTreeOp : public OpKernel{
	public:
		explicit BuildKDTreeOp(OpKernelConstruction* context):OpKernel(context)
		{
      		OP_REQUIRES_OK(context, context->GetAttr("levels", &levels));
		}
		
		void /*std::tuple<T*, int64_t*>*/ checkParamsAndReturnVars(OpKernelContext * context)
		{      
			const Tensor& points_ref_tensor=context->input(0);

			OP_REQUIRES(context,points_ref_tensor.dims()==2,errors::InvalidArgument("BuildKDTree requires xyz1 be of shape (#points,D)"));
			int nr_ref_points = points_ref_tensor.shape().dim_size(0);
            dims = points_ref_tensor.shape().dim_size(1);
			auto points_ref_flat=points_ref_tensor.flat<T>();
			const T * xyz1=&points_ref_flat(0);
			Tensor * structured_points=NULL;
			Tensor * metadata_address=NULL;
			Tensor * shuffle_inds=NULL;
			OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{nr_ref_points, dims}, &structured_points));
			OP_REQUIRES_OK(context,context->allocate_output(1,TensorShape{1}, &metadata_address));
			OP_REQUIRES_OK(context,context->allocate_output(2,TensorShape{nr_ref_points}, &shuffle_inds));
			auto structured_points_flat=structured_points->flat<T>();
			auto metadata_address_flat=metadata_address->flat<point_i_t>(); //auto metadata_address_flat=metadata_address->flat<int64_t>();
			auto shuffle_inds_flat=shuffle_inds->flat<point_i_t>();
			T * structured_points_final=&(structured_points_flat(0));
			point_i_t* meta_addr_final=reinterpret_cast<point_i_t*>(&(metadata_address_flat(0))); //TODO: Data type not supported
			point_i_t* shuffle_inds_final=&(shuffle_inds_flat(0));
			
			vars = /*return*/ std::make_tuple(nr_ref_points, xyz1, structured_points_final, meta_addr_final, shuffle_inds_final);
		}

		void Compute(OpKernelContext * context) override
		{
			/*auto vars =*/ this->checkParamsAndReturnVars(context);
			auto n = std::get<0>(vars);
			auto points_ref = std::get<1>(vars);
            auto structured_point_output = std::get<2>(vars);
			auto metadata_addr = std::get<3>(vars);
			auto shuffle_inds_output = std::get<4>(vars);

            auto& partition_share = ternaryHelper<PartitionShareholder<T>, PartitionShareholder<float>, PartitionShareholder<double>>(partition_share_float, partition_share_double); //std::is_same<T, float>::value ? partition_share_float : partition_share_double;

            //std::cout << levels << ", " << dims << ", " << n << ", " << points_ref << ", " << metadata_addr << ", " << std::endl;
            size_t part_nr;
            T* structured_flat;
			point_i_t* shuffled_inds;


			if(levels > 13) //TODO: Constant
				throw std::runtime_error(std::string("TODO: Maximum number of levels of the KD-Tree currently restricted to ") + std::to_string(13));

			if(dims == 1)
            {
				part_nr = partition_share.addPartition(new PartitionInfo<T, 1>(std::move(createKDTree<T, 1>(points_ref, n, levels))), dims, levels);

                //https://stackoverflow.com/questions/3505713/c-template-compilation-error-expected-primary-expression-before-token
                structured_flat = reinterpret_cast<T*>(partition_share.template getPartition<PartitionInfo<T, 1>*>(part_nr)->structured_points);
				shuffled_inds = partition_share.template getPartition<PartitionInfo<T, 1>*>(part_nr)->shuffled_inds;
            }
			else if(dims == 2)
            {
				part_nr = partition_share.addPartition(new PartitionInfo<T, 2>(std::move(createKDTree<T, 2>(points_ref, n, levels))), dims, levels);
                structured_flat = reinterpret_cast<T*>(partition_share.template getPartition<PartitionInfo<T, 2>*>(part_nr)->structured_points);
				shuffled_inds = partition_share.template getPartition<PartitionInfo<T, 2>*>(part_nr)->shuffled_inds;
            }
			else if(dims == 3)
            {
				part_nr = partition_share.addPartition(new PartitionInfo<T, 3>(std::move(createKDTree<T, 3>(points_ref, n, levels))), dims, levels);
                structured_flat = reinterpret_cast<T*>(partition_share.template getPartition<PartitionInfo<T, 3>*>(part_nr)->structured_points);
				shuffled_inds = partition_share.template getPartition<PartitionInfo<T, 3>*>(part_nr)->shuffled_inds;
            }
			else
				throw std::runtime_error("Unsupported number of dimensions"  + std::to_string(dims)); //TODO: Dynamic implementation//nnsearchDynamic<T, T>(n,m,points_ref,points_query,dist,idx, nr_nns_searches);
			//nnsearch(b,m,n,xyz2,xyz1,dist2,idx2);

			//std::cout << "Shuffled Inds: " << shuffled_inds << std::endl;
            std::copy(structured_flat, structured_flat + n * dims, structured_point_output);
			std::copy(shuffled_inds, shuffled_inds + n, shuffle_inds_output);
            metadata_addr[0] = part_nr;
            //std::cout << "Part Nr: " << part_nr << ", " << partition_share.nrPartitions() << std::endl;
		}

    ~BuildKDTreeOp()
    {
    }

	protected:
		//int nr_nns_searches;
		std::tuple<point_i_t, const T*, T*, point_i_t*, point_i_t*>  vars;
		int levels /*= 5*/;
		int dims /*= 3*/;
        //void* main_partition = NULL;
};

REGISTER_KERNEL_BUILDER(Name("BuildKdTree").Device(DEVICE_CPU).TypeConstraint<float>("T"), BuildKDTreeOp<float>); 
REGISTER_KERNEL_BUILDER(Name("BuildKdTree").Device(DEVICE_CPU).TypeConstraint<double>("T"), BuildKDTreeOp<double>);

template <typename T>
class KDTreeKNNSearchOp : public OpKernel{
	public:
		explicit KDTreeKNNSearchOp(OpKernelConstruction* context) : OpKernel(context)
		{
            //std::cout << "Setting up KDTree Search..." << std::endl;
            OP_REQUIRES_OK(context, context->GetAttr("nr_nns_searches", &nr_nns_searches));
            OP_REQUIRES_OK(context, context->GetAttr("metadata_address_kdtree", &part_nr));
            auto& partition_share = ternaryHelper<PartitionShareholder<T>, PartitionShareholder<float>, PartitionShareholder<double>>(partition_share_float, partition_share_double);
            dims = partition_share.getPartitionDims(part_nr);

            //std::cout << "Fetched partition " << part_nr << " with dimensions " << dims << " and Knns: " << nr_nns_searches << std::endl;
		}
		
		void checkParamsAndReturnVars(OpKernelContext * context){
			const Tensor& points_query_tensor=context->input(0);

			OP_REQUIRES(context,points_query_tensor.dims()==2,errors::InvalidArgument("NnDistance requires points_query be of shape (#points,d)"));
			//OP_REQUIRES(context,points_query_tensor.shape().dim_size(1)==3,errors::InvalidArgument("NnDistance only accepts 3d point set xyz2"));
			int n=points_query_tensor.shape().dim_size(0);
			auto points_query_flat=points_query_tensor.flat<T>();
			const T * points_query=&points_query_flat(0);
			Tensor * dist_tensor=NULL;
			Tensor * idx_tensor=NULL;
			OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{n, nr_nns_searches},&dist_tensor));
			OP_REQUIRES_OK(context,context->allocate_output(1,TensorShape{n, nr_nns_searches},&idx_tensor));
			auto dist_flat=dist_tensor->flat<T>();
			auto idx_flat=idx_tensor->flat<point_i_t>();
			T * dist=&(dist_flat(0));
			point_i_t * idx=&(idx_flat(0));

            if(dims != points_query_tensor.shape().dim_size(1))
                throw std::runtime_error("Error: The KDTree and the required query point cloud differ in dimensionality");
			
			vars = std::make_tuple(n, points_query, dist, idx);
		}
		void Compute(OpKernelContext * context)override{
			
			
			/*auto vars =*/ this->checkParamsAndReturnVars(context);
			auto nr_query_points = std::get<0>(vars);
			auto points_query = std::get<1>(vars);
			auto dist = std::get<2>(vars);
			auto idx_ret = std::get<3>(vars);
            auto knn_idx = idx_ret; //std::vector<point_i_t> knn_idx(n*nr_nns_searches);
            
			
            auto& partition_share = ternaryHelper<PartitionShareholder<T>, PartitionShareholder<float>, PartitionShareholder<double>>(partition_share_float, partition_share_double);
            const auto levels = partition_share.getPartitionLevels(part_nr);

            //std::cout << "Fetched partition " << part_nr << " with levels " << levels << std::endl;

			if(dims == 1)
            {
				KDTreeKNNSearch<T, T, 1>(*partition_share.template getPartition<PartitionInfo<T, 1>*>(part_nr), 
                nr_query_points, reinterpret_cast<const std::array<T, 1>*>(points_query), 
                dist, knn_idx, nr_nns_searches);
            }
			else if(dims == 2)
            {
				KDTreeKNNSearch<T, T, 2>(*partition_share.template getPartition<PartitionInfo<T, 2>*>(part_nr), 
                nr_query_points, reinterpret_cast<const std::array<T, 2>*>(points_query), 
                dist, knn_idx, nr_nns_searches);
            }
			else if(dims == 3)
            {
				KDTreeKNNSearch<T, T, 3>(*partition_share.template getPartition<PartitionInfo<T, 3>*>(part_nr), 
                nr_query_points, reinterpret_cast<const std::array<T, 3>*>(points_query), 
                dist, knn_idx, nr_nns_searches);
            }
			else
				throw std::runtime_error("Unsupported number of dimensions"); //TODO: Dynamic implementation
            

            //std::copy(knn_idx.begin(), knn_idx.end(), idx_ret);
		}

	protected:
		//int nr_nns_searches;
		std::tuple<int, const T*, T*, point_i_t*>  vars;
		int nr_nns_searches = 20;
        int part_nr;
        int dims;
};

REGISTER_KERNEL_BUILDER(Name("KdTreeKnnSearch").Device(DEVICE_CPU).TypeConstraint<float>("T"), KDTreeKNNSearchOp<float>); 
REGISTER_KERNEL_BUILDER(Name("KdTreeKnnSearch").Device(DEVICE_CPU).TypeConstraint<double>("T"), KDTreeKNNSearchOp<double>);

//TODO
//std::map<size_t, T*> mapped_cuda_partition_infos; 

template <typename T>
class KDTreeKNNSearchGPUOp : public KDTreeKNNSearchOp<T>{
	public:
		explicit KDTreeKNNSearchGPUOp(OpKernelConstruction* context) : KDTreeKNNSearchOp<T>(context)
		{
			auto& partition_share = ternaryHelper<PartitionShareholder<T>, PartitionShareholder<float>, PartitionShareholder<double>>(partition_share_float, partition_share_double);

			//std::cout << "Copying partition " << part_nr << std::endl;

			//TODO: Dynamic implementation
			const auto info = partition_share.template getPartition<PartitionInfo<T, 1>*>(part_nr);
			#ifdef DISABLED
			if(info->nr_partitions > max_partitions || info->nr_leaves > max_leaves)
			{
				throw std::runtime_error("Error, please reduce number of levels...");
			}
			#endif

			if(dims == 1)
			{
				partition_info_d = copyPartitionToGPU<T, 1>(*partition_share.template getPartition<PartitionInfo<T, 1>*>(part_nr));
				//partition_info_d_copy = copyPartitionToGPU<T, 1>(*partition_share.template getPartition<PartitionInfo<T, 1>*>(part_nr));
			}
			else if(dims == 2)
			{
				partition_info_d = copyPartitionToGPU<T, 2>(*partition_share.template getPartition<PartitionInfo<T, 2>*>(part_nr));
				//partition_info_d_copy = copyPartitionToGPU<T, 2>(*partition_share.template getPartition<PartitionInfo<T, 2>*>(part_nr));
			}
			else if(dims == 3)
			{
				partition_info_d = copyPartitionToGPU<T, 3>(*partition_share.template getPartition<PartitionInfo<T, 3>*>(part_nr));
				//partition_info_d_copy = copyPartitionToGPU<T, 3>(*partition_share.template getPartition<PartitionInfo<T, 3>*>(part_nr));
			}
			else
				throw std::runtime_error("Unsupported number of dimensions"); //TODO: Dynamic implementation

            //std::cout << "Fetched partition " << part_nr << " with dimensions " << dims << " and Knns: " << nr_nns_searches << std::endl;
		}

		~KDTreeKNNSearchGPUOp()
		{
			if(dims == 1)
			{
				freePartitionFromGPU(reinterpret_cast<PartitionInfoDevice<T, 1>*>(partition_info_d));
				//freePartitionFromGPU(reinterpret_cast<PartitionInfoDevice<T, 1>*>(partition_info_d_copy));
			}
			else if(dims == 2)
			{
				freePartitionFromGPU(reinterpret_cast<PartitionInfoDevice<T, 2>*>(partition_info_d));
				//freePartitionFromGPU(reinterpret_cast<PartitionInfoDevice<T, 2>*>(partition_info_d_copy));
			}
			else if(dims == 3)
			{
				freePartitionFromGPU(reinterpret_cast<PartitionInfoDevice<T, 3>*>(partition_info_d));
				//freePartitionFromGPU(reinterpret_cast<PartitionInfoDevice<T, 3>*>(partition_info_d_copy));
			}
		}
		
		void Compute(OpKernelContext * context) override
		{			
			/*auto vars =*/ this->checkParamsAndReturnVars(context);
			auto nr_query_points = std::get<0>(vars);
			auto points_query = std::get<1>(vars);
			auto dist = std::get<2>(vars);
			auto idx_ret = std::get<3>(vars);
            auto knn_idx = idx_ret; //std::vector<point_i_t> knn_idx(n*nr_nns_searches);
            
			
            auto& partition_share = ternaryHelper<PartitionShareholder<T>, PartitionShareholder<float>, PartitionShareholder<double>>(partition_share_float, partition_share_double);
            const auto levels = partition_share.getPartitionLevels(part_nr);

            //std::cout << "Fetched partition " << part_nr << " with levels " << levels << std::endl;

			if(dims == 1)
            {
				KDTreeKNNGPUSearch<T, T, 1>(reinterpret_cast<PartitionInfoDevice<T, 1>*>(partition_info_d), 
                nr_query_points, reinterpret_cast<const std::array<T, 1>*>(points_query), 
                dist, knn_idx, nr_nns_searches);
            }
			else if(dims == 2)
            {
				KDTreeKNNGPUSearch<T, T, 2>(reinterpret_cast<PartitionInfoDevice<T, 2>*>(partition_info_d), 
                nr_query_points, reinterpret_cast<const std::array<T, 2>*>(points_query), 
                dist, knn_idx, nr_nns_searches);
            }
			else if(dims == 3)
            {
				KDTreeKNNGPUSearch<T, T, 3>(reinterpret_cast<PartitionInfoDevice<T, 3>*>(partition_info_d), 
                nr_query_points, reinterpret_cast<const std::array<T, 3>*>(points_query), 
                dist, knn_idx, nr_nns_searches);
            }
			else
				throw std::runtime_error("Unsupported number of dimensions"); //TODO: Dynamic implementation
            

            //std::copy(knn_idx.begin(), knn_idx.end(), idx_ret);
		}

	protected:
		//int nr_nns_searches;
		using KDTreeKNNSearchOp<T>::vars;
		using KDTreeKNNSearchOp<T>::nr_nns_searches;
        using KDTreeKNNSearchOp<T>::part_nr;
        using KDTreeKNNSearchOp<T>::dims;
		//PartitionInfo<T, dims>* selected_partition;
		void* partition_info_d;
		//void* partition_info_d_copy;
};

REGISTER_KERNEL_BUILDER(Name("KdTreeKnnSearch").Device(DEVICE_GPU).TypeConstraint<float>("T"), KDTreeKNNSearchGPUOp<float>); 
REGISTER_KERNEL_BUILDER(Name("KdTreeKnnSearch").Device(DEVICE_GPU).TypeConstraint<double>("T"), KDTreeKNNSearchGPUOp<double>);
