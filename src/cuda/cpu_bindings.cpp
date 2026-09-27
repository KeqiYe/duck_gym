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
#include "kernel.cuh"
#undef DUCK_NAMESPACE
#undef DUCK_SCALAR
#define DUCK_NAMESPACE cpu64
#define DUCK_SCALAR double
#include "kernel.cuh"
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
          s.joints[j].torque = model.joints[j].torque;
          s.joints[j].frictionloss = model.joints[j].frictionloss;
          s.joints[j].damping = model.joints[j].damping;
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

  void stepMotor(A motor, A forces) {
    check(motor, {count, model.joints.size(), 3});
    check(forces, {count, 3});
    // Single-step interface: the caller must recompute velocity/load-dependent
    // actuator output at each physical step, never once per control interval.
    for (int e = 0; e < count; ++e) {
      auto &s = states[e];
      s.setMotor(motor.data() + e * model.joints.size() * 3,
                 forces.data() + e * 3);
      if (!s.failed)
        s.advance(1);
    }
  }
  A contactForces(bool groundOnly) {
    A out({count, model.bodies.size(), 3});
    for (int e = 0; e < count; ++e)
      states[e].contactForces(out.mutable_data() + e * model.bodies.size() * 3,
                              groundOnly);
    return out;
  }
  A selfContactStats() {
    A out({count, 2});
    for (int e = 0; e < count; ++e)
      states[e].selfContactStats(out.mutable_data() + 2 * e);
    return out;
  }
  A contactCounts() {
    A out({count, 2});
    for (int e = 0; e < count; ++e)
      states[e].contactCounts(out.mutable_data() + 2 * e);
    return out;
  }
  A generalizedLoads(A rootOffset) {
    check(rootOffset, {count, 3});
    A out({count, model.joints.size(), 3});
    for (int e = 0; e < count; ++e) {
      auto p = rootOffset.data() + e * 3;
      for (int k = 0; k < 3; ++k)
        duck_native::require(std::isfinite(p[k]), "Invalid root COM offset");
      states[e].generalizedLoads(
          {p[0], p[1], p[2]}, out.mutable_data() + e * model.joints.size() * 3);
    }
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
      .def("step_motor", &B::stepMotor)
      .def("state", &B::state)
      .def("contact_forces", &B::contactForces, py::arg("ground_only") = false)
      .def("self_contact_stats", &B::selfContactStats)
      .def("contact_counts", &B::contactCounts)
      .def("generalized_loads", &B::generalizedLoads)
      .def_property_readonly("joint_names", &B::joint_names);
}

// Diagnostic access to the same bounded MPR narrow phase used by the solver.
template <bool fp64>
py::dict convexQuery(
    py::array_t<double, py::array::c_style | py::array::forcecast> va,
    py::array_t<double, py::array::c_style | py::array::forcecast> vb,
    py::array_t<double, py::array::c_style | py::array::forcecast> poses,
    double inflation, double tolerance) {
  using V = std::conditional_t<fp64, cpu64::Vec3, cpu32::Vec3>;
  using B = std::conditional_t<fp64, cpu64::Body, cpu32::Body>;
  using Sh = std::conditional_t<fp64, cpu64::Shape, cpu32::Shape>;
  using Q =
      std::conditional_t<fp64, cpu64::MinkowskiQuery, cpu32::MinkowskiQuery>;
  using R = std::conditional_t<fp64, double, float>;
  duck_native::require(poses.ndim() == 2 && poses.shape(0) == 2 &&
                           poses.shape(1) == 7,
                       "poses must be (2,7): xyz,wxyz");
  duck_native::require(inflation >= 0 && std::isfinite(inflation) &&
                           tolerance > 0 && std::isfinite(tolerance),
                       "Invalid collision tolerance/inflation");
  std::vector<V> points;
  B bodies[2];
  Sh shapes[2];
  for (int k = 0; k < 2; ++k) {
    auto &v = k == 0 ? va : vb;
    auto &b = bodies[k];
    auto &s = shapes[k];
    duck_native::require(v.ndim() == 2 && v.shape(0) >= 4 && v.shape(1) == 3,
                         "Convex vertices must be (N>=4,3)");
    s.type = 3;
    s.offset = points.size();
    s.count = v.shape(0);
    for (int i = 0; i < s.count; ++i) {
      V x;
      for (int j = 0; j < 3; ++j) {
        duck_native::require(std::isfinite(v.data()[i * 3 + j]),
                             "Nonfinite vertex");
        x[j] = v.data()[i * 3 + j];
      }
      points.push_back(x);
      s.interior += x / R(s.count);
      s.boundingRadius = std::max(s.boundingRadius, norm(x));
    }
    const double *p = poses.data() + 7 * k;
    for (int j = 0; j < 7; ++j)
      duck_native::require(std::isfinite(p[j]), "Nonfinite pose");
    duck_native::require(p[3] * p[3] + p[4] * p[4] + p[5] * p[5] + p[6] * p[6] >
                             1e-12,
                         "Zero quaternion");
    b.x = {R(p[0]), R(p[1]), R(p[2])};
    b.q = normalized(decltype(b.q){R(p[3]), R(p[4]), R(p[5]), R(p[6])});
  }
  auto c = convexContact(Q{shapes[0], shapes[1], bodies[0], bodies[1],
                           points.data(), R(inflation)},
                         R(tolerance));
  py::dict out;
  out["status"] = c.status;
  out["depth"] = c.depth;
  out["normal"] = py::make_tuple(c.normal.x, c.normal.y, c.normal.z);
  out["position"] = py::make_tuple(c.position.x, c.position.y, c.position.z);
  return out;
}

PYBIND11_MODULE(_duck_reference, m) {
  m.def("convex_query64", &convexQuery<true>, py::arg("vertices_a"),
        py::arg("vertices_b"), py::arg("poses"), py::arg("inflation") = 0.,
        py::arg("tolerance") = 1e-8);
  m.def("convex_query32", &convexQuery<false>, py::arg("vertices_a"),
        py::arg("vertices_b"), py::arg("poses"), py::arg("inflation") = 0.,
        py::arg("tolerance") = 1e-6);
  bind<cpu32::State>(m, "CpuBatch32");
  bind<cpu64::State>(m, "CpuBatch64");
}
