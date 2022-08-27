#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <cmath>
#include <iostream>
#include <vector>

#include "xgboost/xgboost.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

using namespace std;
namespace py = pybind11;

template <typename... Args>
using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;

PYBIND11_MODULE(aijack_secureboost, m)
{
    m.doc() = R"pbdoc(
        core of XGBoost
    )pbdoc";

    py::class_<XGBoostParty>(m, "XGBoostParty")
        .def(py::init<vector<vector<float>>, int, vector<int>, int,
                      int, float, int, bool, int>())
        .def("get_lookup_table", &XGBoostParty::get_lookup_table);

    py::class_<XGBoostNode>(m, "XGBoostNode")
        .def("get_idxs", &XGBoostNode::get_idxs)
        .def("get_party_id", &XGBoostNode::get_party_id)
        .def("get_record_id", &XGBoostNode::get_record_id)
        .def("get_val", &XGBoostNode::get_val)
        .def("get_score", &XGBoostNode::get_score)
        .def("get_left", &XGBoostNode::get_left)
        .def("get_right", &XGBoostNode::get_right)
        .def("is_leaf", &XGBoostNode::is_leaf);

    py::class_<XGBoostTree>(m, "XGBoostTree")
        .def("get_root_xgboost_node", &XGBoostTree::get_root_xgboost_node);

    py::class_<XGBoostClassifier>(m, "XGBoostClassifier")
        .def(py::init<double, double, int, int, double, int, double, double, double>())
        .def("fit", &XGBoostClassifier::fit)
        .def("get_init_pred", &XGBoostClassifier::get_init_pred)
        .def("load_estimators", &XGBoostClassifier::load_estimators)
        .def("get_estimators", &XGBoostClassifier::get_estimators)
        .def("predict_raw", &XGBoostClassifier::predict_raw)
        .def("predict_proba", &XGBoostClassifier::predict_proba);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
