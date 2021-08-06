#include <pybind11/pybind11.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

int add(int i, int j) {
    return i + j;
}

namespace py = pybind11;

//py::array_t<double, py::array::c_style | py::array::forcecast>

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

#ifdef VERSION_INFO //Set by cmake
    mod.attr("__version__") = VERSION_INFO; //MACRO_STRINGIFY(VERSION_INFO);
#else
    mod.attr("__version__") = "dev";
#endif
}
