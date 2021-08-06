#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "kdtree.hpp"
#include "kdtree_g.hpp"


#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)


namespace py = pybind11;
/*template <typename T>
class PartitionShareholder
{
public:
    ~PartitionShareholder()
    {
        size_t part_i = 0;
        for (part_i = 0; part_i < partition_infos.size(); part_i++)
        {
            const auto dims = dimensions[part_i];
            const auto main_partition = partition_infos[part_i];

            if (dims == 1)
                delete reinterpret_cast<PartitionInfo<T, 1>*>(main_partition);
            else if (dims == 2)
                delete reinterpret_cast<PartitionInfo<T, 2>*>(main_partition);
            else if (dims == 3)
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
        if (part_nr > partition_infos.size())
            throw std::runtime_error("You requested a non-present partition... This probably happened because you used a different precision for the KDtree, multiple GPUs or machines, which this library does not currently support");

        return reinterpret_cast<Ret>(partition_infos[part_nr]);
    }

    dim_t getPartitionDims(const size_t part_nr) const
    {
        if (part_nr > partition_infos.size())
            throw std::runtime_error("You requested a non-present partition... This probably happened because you used a different precision for the KDtree, multiple GPUs or machines, which this library does not currently support");

        return dimensions[part_nr];
    }

    int getPartitionLevels(const size_t part_nr) const
    {
        if (part_nr > partition_infos.size())
            throw std::runtime_error("You requested a non-present partition... This probably happened because you used a different precision for the KDtree, multiple GPUs or machines, which this library does not currently support");

        return partition_levels[part_nr];
    }

    size_t nrPartitions() const { return partition_infos.size(); }

protected:
    std::vector<void*> partition_infos;
    std::vector<dim_t> dimensions;
    std::vector<int> partition_levels;
};

static PartitionShareholder<float> partition_share_float;
static PartitionShareholder<double> partition_share_double;*/

template <typename T, dim_t dims, bool use_gpu>
struct KDTree
{
    int levels;
    PartitionInfo<T, dims>* partition_info;
    PartitionInfoDevice<T, dims>* partition_info_d;

    //For repeated queries, local buffers will avoid reallocation of the necessary buffers
    /*T* dist_buf = NULL;
    point_i_t* knn_idx_buf = NULL;
    point_i_knn_t nr_nns_buf = 0;*/

    KDTree(const py::array_t<T, py::array::c_style | py::array::forcecast>& points_ref, const int levels)
    {
        const auto dims_arr = points_ref.shape(1);
        const float* points_raw = points_ref.data();
        const auto nr_points = points_ref.shape(0);

        {
            py::gil_scoped_release release;
            this->levels = levels;
            assert(dims_arr == dims);

            partition_info = new PartitionInfo<T, dims>(std::move(createKDTree<T, dims>(points_raw, nr_points, levels)));

            if (use_gpu)
            {
                partition_info_d = copyPartitionToGPU<T, dims>(*partition_info);
            }
        }
    }

    py::array_t<point_i_t> get_shuffled_inds() const
    {
        const auto shuffled_inds = partition_info->shuffled_inds;
        auto shuffled_inds_arr = py::array_t<point_i_t>(partition_info->nr_points, shuffled_inds);
        return std::move(shuffled_inds_arr);
    }

    py::array_t<T> get_structured_points() const
    {
        const auto structured_points = partition_info->structured_points;
        auto structured_points_arr = py::array_t<T>(partition_info->nr_points * dims, structured_points->data());
        structured_points_arr.resize(std::vector<ptrdiff_t>{ partition_info->nr_points, dims });
        return std::move(structured_points_arr);
    }

    void query_recast(const T* points_query, const size_t nr_query_points, const point_i_knn_t nr_nns_searches, T* dist_arr, point_i_knn_t* knn_idx)
    {
        {
            py::gil_scoped_release release;
            if (use_gpu)
            {
                KDTreeKNNGPUSearch<T, T, dims>(partition_info_d,
                    nr_query_points, reinterpret_cast<const std::array<T, dims>*>(points_query),
                    dist_arr, knn_idx, nr_nns_searches);
            }
            else
            {
                KDTreeKNNSearch<T, T, dims>(*partition_info,
                    nr_query_points, reinterpret_cast<const std::array<T, dims>*>(points_query),
                    dist_arr, knn_idx, nr_nns_searches);
            }
        }
    }

    void query(const size_t points_query_ptr, const size_t nr_query_points, const point_i_knn_t nr_nns_searches, const size_t dist_arr_ptr, const size_t knn_idx_ptr)
    {
        //Necessary for CUDA raw pointers being passed around. They can NOT be converted to a py::array_t
        T* points_query = reinterpret_cast<T*>(points_query_ptr);
        T* dist_arr = reinterpret_cast<T*>(dist_arr_ptr);
        point_i_knn_t* knn_idx = reinterpret_cast<point_i_knn_t*>(knn_idx_ptr);
        this->query_recast(points_query, nr_query_points, nr_nns_searches, dist_arr, knn_idx);
    }

    ~KDTree()
    {
        delete partition_info;

        if (use_gpu)
            freePartitionFromGPU(partition_info_d);
    }
};

int add(int i, int j)
{
    return i + j;
}


PYBIND11_MODULE(cp_knn, mod) {
    mod.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: cp_knn

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    py::class_< KDTree<float, 3, false>, std::shared_ptr< KDTree<float, 3, false>>>(mod, "KDTreeCPU3DF")
        .def(py::init<py::array_t<float>, int>(), py::arg("points_ref"), py::arg("levels"))
        .def("get_shuffled_inds", &KDTree<float, 3, false>::get_shuffled_inds)
        .def("get_structured_points", &KDTree<float, 3, false>::get_structured_points)
        .def("query", &KDTree<float, 3, false>::query);

    py::class_< KDTree<float, 3, true>, std::shared_ptr< KDTree<float, 3, true>>>(mod, "KDTreeGPU3DF")
        .def(py::init<py::array_t<float>, int>(), py::arg("points_ref"), py::arg("levels"))
        .def("get_shuffled_inds", &KDTree<float, 3, true>::get_shuffled_inds)
        .def("get_structured_points", &KDTree<float, 3, true>::get_structured_points)
        .def("query", &KDTree<float, 3, true>::query);

    mod.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

    mod.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");

    //mod.def("buildKDTreeCPU", buildKDTreeCPU<double>);
    /*mod.def("buildKDTreeCPUF", buildKDTreeCPUF,
            py::arg("points_ref"), py::arg("levels"));

    mod.def("buildKDTreeGPUF", buildKDTreeGPUF,
        py::arg("points_ref"), py::arg("levels"));

    mod.def("searchKDTreeCPUFA", searchKDTreeCPUFA,
            py::arg("points_query"), py::arg("nr_nns_searches"), py::arg("part_nr"));*/

#ifdef VERSION_INFO //Set by cmake
    mod.attr("__version__") = VERSION_INFO; //MACRO_STRINGIFY(VERSION_INFO);
#else
    mod.attr("__version__") = "dev";
#endif
}
