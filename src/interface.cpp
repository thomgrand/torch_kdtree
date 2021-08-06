#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "kdtree.hpp"


#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)


namespace py = pybind11;
template <typename T>
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
static PartitionShareholder<double> partition_share_double;

//py::array_t<double, py::array::c_style | py::array::forcecast>
py::tuple buildKDTreeCPUF(const py::array_t<float, py::array::c_style | py::array::forcecast>& points_ref, const int levels)
{
    //const auto points_raw = points_ref.unchecked<2>(); //Raw pointer of dimensionality 2
    const auto dims = points_ref.shape(1);
    const float* points_raw = points_ref.data();
    const auto nr_points = points_ref.shape(0);

    auto& partition_share = partition_share_float;

    size_t part_nr;
    float* structured_flat;
    point_i_t* shuffled_inds;

    if (dims == 1)
    {
        part_nr = partition_share.addPartition(new PartitionInfo<float, 1>(std::move(createKDTree<float, 1>(points_raw, nr_points, levels))), dims, levels);

        //https://stackoverflow.com/questions/3505713/c-template-compilation-error-expected-primary-expression-before-token
        structured_flat = reinterpret_cast<float*>(partition_share.template getPartition<PartitionInfo<float, 1>*>(part_nr)->structured_points);
        shuffled_inds = partition_share.template getPartition<PartitionInfo<float, 1>*>(part_nr)->shuffled_inds;
    }
    else if (dims == 2)
    {
        part_nr = partition_share.addPartition(new PartitionInfo<float, 2>(std::move(createKDTree<float, 2>(points_raw, nr_points, levels))), dims, levels);
        structured_flat = reinterpret_cast<float*>(partition_share.template getPartition<PartitionInfo<float, 2>*>(part_nr)->structured_points);
        shuffled_inds = partition_share.template getPartition<PartitionInfo<float, 2>*>(part_nr)->shuffled_inds;
    }
    else if (dims == 3)
    {
        part_nr = partition_share.addPartition(new PartitionInfo<float, 3>(std::move(createKDTree<float, 3>(points_raw, nr_points, levels))), dims, levels);
        structured_flat = reinterpret_cast<float*>(partition_share.template getPartition<PartitionInfo<float, 3>*>(part_nr)->structured_points);
        shuffled_inds = partition_share.template getPartition<PartitionInfo<float, 3>*>(part_nr)->shuffled_inds;
    }
    else
        throw std::runtime_error("Unsupported number of dimensions" + std::to_string(dims)); //TODO: Dynamic implementation

    //Result arrays
    /* No pointer is passed, so NumPy will allocate the buffer */
    //According to some issues on github, this should copy the data
    //auto shuffled_inds_arr = py::array_t<point_i_t>(nr_points);
    //auto structured_points_arr = py::array_t<float>(points_ref.size());
    auto shuffled_inds_arr = py::array_t<point_i_t>(nr_points, shuffled_inds);
    auto structured_points_arr = py::array_t<float>(points_ref.size(), structured_flat); //Pass the array directly to numpy for use in python


    return py::make_tuple(structured_points_arr, part_nr, shuffled_inds_arr);
}

py::tuple searchKDTreeCPUF(const py::array_t<float, py::array::c_style | py::array::forcecast>& points_query,
    const point_i_knn_t nr_nns_searches,
    const size_t part_nr,
    py::array_t<float>& dist_arr, py::array_t<point_i_t>& knn_idx_arr)
{
    //const auto points_raw = points_ref.unchecked<2>(); //Raw pointer of dimensionality 2
    //const auto dims = points_query.shape(1);
    const float* points_raw = points_query.data();
    const auto nr_query_points = points_query.shape(0);

    auto& partition_share = partition_share_float;
    const auto dims = partition_share.getPartitionDims(part_nr);

    if (dims != points_query.shape(1))
        throw std::runtime_error("Error: The KDTree and the required query point cloud differ in dimensionality");

    const auto levels = partition_share.getPartitionLevels(part_nr);

    //Result arrays
    /* No pointer is passed, so NumPy will allocate the buffer */
    auto dist = dist_arr.mutable_data();
    auto knn_idx = knn_idx_arr.mutable_data();

    if (dims == 1)
    {
        KDTreeKNNSearch<float, float, 1>(*partition_share.template getPartition<PartitionInfo<float, 1>*>(part_nr),
            nr_query_points, reinterpret_cast<const std::array<float, 1>*>(points_raw),
            dist, knn_idx, nr_nns_searches);
    }
    else if (dims == 2)
    {
        KDTreeKNNSearch<float, float, 2>(*partition_share.template getPartition<PartitionInfo<float, 2>*>(part_nr),
            nr_query_points, reinterpret_cast<const std::array<float, 2>*>(points_raw),
            dist, knn_idx, nr_nns_searches);
    }
    else if (dims == 3)
    {
        KDTreeKNNSearch<float, float, 3>(*partition_share.template getPartition<PartitionInfo<float, 3>*>(part_nr),
            nr_query_points, reinterpret_cast<const std::array<float, 3>*>(points_raw),
            dist, knn_idx, nr_nns_searches);
    }
    else
        throw std::runtime_error("Unsupported number of dimensions"); //TODO: Dynamic implementation

    return py::make_tuple(dist_arr, knn_idx_arr);
}

py::tuple searchKDTreeCPUFA(const py::array_t<float, py::array::c_style | py::array::forcecast>& points_query,
                            const point_i_knn_t nr_nns_searches,
                            const size_t part_nr)
{
    //const auto points_raw = points_ref.unchecked<2>(); //Raw pointer of dimensionality 2
    //const auto dims = points_query.shape(1);
    const float* points_raw = points_query.data();
    const auto nr_query_points = points_query.shape(0);

    //Result arrays
    /* No pointer is passed, so NumPy will allocate the buffer */
    auto dist_arr = py::array_t<float>(nr_query_points * nr_nns_searches);
    auto knn_idx_arr = py::array_t<point_i_t>(nr_query_points * nr_nns_searches);
    dist_arr.resize(std::vector<ptrdiff_t>{ nr_query_points, nr_nns_searches });
    knn_idx_arr.resize(std::vector<ptrdiff_t>{ nr_query_points, nr_nns_searches });

    return searchKDTreeCPUF(points_query, nr_nns_searches, part_nr, dist_arr, knn_idx_arr);
}

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

    mod.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

    mod.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");

    //mod.def("buildKDTreeCPU", buildKDTreeCPU<double>);
    mod.def("buildKDTreeCPUF", buildKDTreeCPUF,
            py::arg("points_ref"), py::arg("levels"));

    mod.def("searchKDTreeCPUFA", searchKDTreeCPUFA,
            py::arg("points_query"), py::arg("nr_nns_searches"), py::arg("part_nr"));

#ifdef VERSION_INFO //Set by cmake
    mod.attr("__version__") = VERSION_INFO; //MACRO_STRINGIFY(VERSION_INFO);
#else
    mod.attr("__version__") = "dev";
#endif
}
