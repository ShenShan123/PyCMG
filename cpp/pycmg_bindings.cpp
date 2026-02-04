#include <pybind11/pybind11.h>

namespace py = pybind11;

struct PycmgModelPlaceholder {};

PYBIND11_MODULE(_pycmg, m) {
  m.doc() = "pycmg OSDI bindings";
  py::class_<PycmgModelPlaceholder>(m, "Model");
}
