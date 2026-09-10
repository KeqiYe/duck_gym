// Portable CPU instantiation of the exact CUDA solver math and color order.
#include "model.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <thread>
#define __host__
#define __device__
#define DUCK_NAMESPACE cpu32
#define DUCK_SCALAR float
#include "kernel.inl"
#undef DUCK_NAMESPACE
#undef DUCK_SCALAR
#define DUCK_NAMESPACE cpu64
#define DUCK_SCALAR double
#include "kernel.inl"
#undef __host__
#undef __device__
namespace py = pybind11;
template <class S> class CpuBatch {
  using Real = typename S::Scalar;
  using A = py::array_t<Real, py::array::c_style>;
  typename S::ModelType model;
  std::vector<typename S::Point> points;
  std::vector<S> states;
  std::vector<std::string> names;
  int count, workers;
  void check(const py::array &a, std::vector<py::ssize_t> shape) {
    duck_native::require(a.ndim() == int(shape.size()), "Invalid array rank");
    for (int k = 0; k < a.ndim(); ++k)
      duck_native::require(a.shape(k) == shape[k], "Invalid array shape");
  }

public:
  CpuBatch(std::string path, int n, double dt, int iterations, int threads)
      : count(n), workers(std::min(n, threads)) {
    duck_native::require(n > 0 && n <= 4096 && dt > 0 && std::isfinite(dt) &&
                             iterations > 0 && threads > 0 && threads <= 64,
                         "Invalid batch parameters");
    duck::Solver original;
    original.load(path);
    duck_native::convert<S>(original, model, points, dt, iterations);
    for (auto &j : original.joints)
      names.push_back(j.name);
    states.resize(n);
    for (auto &s : states)
      s.initialize(&model, points.data());
  }
  void reset(py::array_t<bool, py::array::c_style> mask, A q, A root) {
    check(mask, {count});
    check(q, {count, model.joints.size()});
    check(root, {count, 6});
    for (int e = 0; e < count; ++e)
      if (mask.data()[e]) {
        auto &s = states[e];
        bool valid = true;
        for (int j = 0; j < model.joints.size(); ++j)
          valid &= std::isfinite(q.data()[e * model.joints.size() + j]);
        for (int j = 0; j < 6; ++j)
          valid &= std::isfinite(root.data()[e * 6 + j]);
        if (!valid) {
          s.failed = 1;
          continue;
        }
        s.initialize(&model, points.data());
        s.resetPose(q.data() + e * model.joints.size(), root.data() + e * 6);
      }
  }
  void step(A target, A forces, int substeps, double kp, double limit) {
    check(target, {count, model.joints.size()});
    check(forces, {count, 3});
    duck_native::require(substeps > 0 && kp >= 0 && std::isfinite(kp) &&
                             limit > 0 && std::isfinite(limit),
                         "Invalid controls");
    const Real *targets = target.data(), *external = forces.data();
    auto work = [&](int worker) {
      for (int e = worker; e < count; e += workers) {
        auto &s = states[e];
        for (int j = 0; j < s.joints.size(); ++j) {
          auto t = targets[e * s.joints.size() + j];
          if (!std::isfinite(t))
            s.failed = 1;
          s.joints[j].target = t;
          s.joints[j].kp = kp;
          s.joints[j].maxTorque = limit;
        }
        for (int k = 0; k < 3; ++k)
          if (!std::isfinite(external[e * 3 + k]))
            s.failed = 1;
        if (s.failed)
          continue;
        s.bodies[1].force = {external[e * 3], external[e * 3 + 1],
                             external[e * 3 + 2]};
        s.advance(substeps);
      }
    };
    std::vector<std::thread> threads;
    try {
      for (int w = 1; w < workers; ++w)
        threads.emplace_back(work, w);
      work(0);
    } catch (...) {
      for (auto &thread : threads)
        thread.join();
      throw;
    }
    for (auto &thread : threads)
      thread.join();
  }

  A contactForces() {
    A out({count, model.bodies.size(), 3});
    for (int e = 0; e < count; ++e)
      states[e].contactForces(out.mutable_data() + e * model.bodies.size() * 3);
    return out;
  }
  py::tuple state() {
    A b({count, model.bodies.size(), 13}), j({count, model.joints.size(), 2}),
        d({count, 6});
    for (int e = 0; e < count; ++e)
      states[e].snapshot(b.mutable_data() + e * model.bodies.size() * 13,
                         j.mutable_data() + e * model.joints.size() * 2,
                         d.mutable_data() + e * 6);
    return py::make_tuple(b, j, d);
  }
  std::vector<std::string> joint_names() const { return names; }
};
template <class S> void bind(py::module_ &m, const char *name) {
  using B = CpuBatch<S>;
  py::class_<B>(m, name)
      .def(py::init<std::string, int, double, int, int>(), py::arg("model"),
           py::arg("num_envs"), py::arg("dt"), py::arg("iterations"),
           py::arg("threads") = 1)
      .def("reset", &B::reset)
      .def("step", &B::step)
      .def("state", &B::state)
      .def("contact_forces", &B::contactForces)
      .def_property_readonly("joint_names", &B::joint_names);
}
PYBIND11_MODULE(_duck_reference, m) {
  bind<cpu32::State>(m, "CpuBatch32");
  bind<cpu64::State>(m, "CpuBatch64");
}
