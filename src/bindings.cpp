#include "duck/solver.hpp"
#include <exception>
#include <mutex>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <thread>
namespace py = pybind11;
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

class CpuBatch {
  duck::Solver prototype;
  std::vector<duck::Solver> envs;
  std::vector<duck::Diagnostics> controlDiagnostics;
  int workers;
  mutable std::mutex mutex;
  std::unique_lock<std::mutex> lock() const {
    std::unique_lock<std::mutex> guard(mutex, std::try_to_lock);
    if (!guard.owns_lock())
      throw std::runtime_error("CpuBatch is already in use by another caller");
    return guard;
  }
  static void finite(const Array &a) {
    for (py::ssize_t i = 0; i < a.size(); ++i)
      if (!std::isfinite(a.data()[i]))
        throw std::invalid_argument("Non-finite native input");
  }

public:
  CpuBatch(const std::string &path, int count, double dt, int iterations,
           int threads)
      : workers(threads) {
    if (count < 1 || count > 4096 || threads < 1 || !std::isfinite(dt) ||
        dt <= 0 || iterations < 1)
      throw std::invalid_argument("Invalid environment/thread count");
    prototype.load(path);
    prototype.options.dt = dt;
    prototype.options.iterations = iterations;
    if (prototype.bodies.size() < 2 || prototype.bodies[1].mass <= 0)
      throw std::invalid_argument(
          "CpuBatch requires a floating root at body 1");
    std::vector<bool> seen(prototype.bodies.size());
    seen[0] = seen[1] = true;
    for (auto &j : prototype.joints) {
      if (!seen[j.a] || seen[j.b])
        throw std::invalid_argument(
            "Expected topologically ordered floating articulation");
      seen[j.b] = true;
    }
    for (bool v : seen)
      if (!v)
        throw std::invalid_argument("Disconnected body");
    envs.assign(count, prototype);
    controlDiagnostics.resize(count);
  }
  int num_envs() const { return (int)envs.size(); }
  int num_joints() const { return (int)prototype.joints.size(); }
  int num_bodies() const { return (int)prototype.bodies.size(); }
  std::vector<std::string> joint_names() const {
    std::vector<std::string> out;
    for (auto &j : prototype.joints)
      out.push_back(j.name);
    return out;
  }
  void reset(const std::vector<int> &ids, Array angles, Array roots) {
    auto guard = lock();
    finite(angles);
    finite(roots);
    for (auto id : ids)
      if (id < 0 || id >= num_envs())
        throw std::out_of_range("Invalid environment id");
    auto q = angles.unchecked<2>();
    auto r = roots.unchecked<2>();
    if (q.shape(0) != (int)ids.size() || q.shape(1) != num_joints() ||
        r.shape(0) != (int)ids.size() || r.shape(1) != 6)
      throw std::invalid_argument(
          "Reset expects (ids,joints) and root xyz/rotation-vector (ids,6)");
    for (int n = 0; n < (int)ids.size(); ++n) {
      auto &s = envs.at(ids[n]);
      s.reset();
      for (auto &body : s.bodies) {
        body.v = {};
        body.w = {};
        body.force = {};
      }
      controlDiagnostics[ids[n]] = {};
      s.bodies[1].x += duck::Vec3{r(n, 0), r(n, 1), r(n, 2)};
      s.bodies[1].q = duck::exp({r(n, 3), r(n, 4), r(n, 5)}) * s.bodies[1].q;
      for (int j = 0; j < num_joints(); ++j) {
        auto &joint = s.joints[j];
        const auto &old = prototype.bodies[joint.a];
        auto &a = s.bodies[joint.a];
        auto &b = s.bodies[joint.b];
        double delta = q(n, j) - prototype.angle(prototype.joints[j]);
        duck::Vec3 axis = duck::rotate(a.q, joint.axisA);
        b.q = duck::normalized(duck::exp(axis * delta) * a.q *
                               duck::conjugate(old.q) *
                               prototype.bodies[joint.b].q);
        b.x = a.x + duck::rotate(a.q, joint.ra) - duck::rotate(b.q, joint.rb);
      }
    }
  }
  void step(Array targets, int substeps, double kp, double torque_limit,
            Array forces) {
    auto guard = lock();
    finite(targets);
    finite(forces);
    auto q = targets.unchecked<2>();
    auto f = forces.unchecked<2>();
    if (q.shape(0) != num_envs() || q.shape(1) != num_joints() ||
        f.shape(0) != num_envs() || f.shape(1) != 3 || substeps < 1 ||
        !std::isfinite(kp) || kp < 0 || !std::isfinite(torque_limit) ||
        torque_limit <= 0)
      throw std::invalid_argument("Invalid step shapes/control parameters");
    for (int i = 0; i < num_envs(); ++i) {
      envs[i].bodies[1].force = {f(i, 0), f(i, 1), f(i, 2)};
      for (int j = 0; j < num_joints(); ++j) {
        if (!std::isfinite(q(i, j)))
          throw std::invalid_argument("Non-finite action");
        envs[i].joints[j].target = q(i, j);
        envs[i].joints[j].kp = kp;
        envs[i].joints[j].maxTorque = torque_limit;
      }
    }
    py::gil_scoped_release release;
    std::vector<std::exception_ptr> errors(num_envs());
    auto run = [&](int first, int stride) {
      for (int i = first; i < num_envs(); i += stride) {
        try {
          auto &summary = controlDiagnostics[i];
          summary = {};
          summary.converged = true;
          for (int k = 0; k < substeps; ++k) {
            envs[i].step();
            const auto &d = envs[i].diagnostics;
            summary.jointError = std::max(summary.jointError, d.jointError);
            summary.axisError = std::max(summary.axisError, d.axisError);
            summary.penetration = std::max(summary.penetration, d.penetration);
            summary.contacts = std::max(summary.contacts, d.contacts);
            summary.converged = summary.converged && d.converged;
          }
        } catch (...) {
          errors[i] = std::current_exception();
        }
      }
    };
    int n = std::min(workers, num_envs());
    std::vector<std::thread> threads;
    struct JoinThreads {
      std::vector<std::thread> &threads;
      ~JoinThreads() {
        for (auto &t : threads)
          if (t.joinable())
            t.join();
      }
    } joinOnException{threads};
    for (int i = 1; i < n; ++i)
      threads.emplace_back(run, i, n);
    run(0, n);
    for (auto &t : threads)
      t.join();
    for (auto e : errors)
      if (e)
        std::rethrow_exception(e);
  }
  Array body_state() const {
    auto guard = lock();
    Array out({num_envs(), num_bodies(), 13});
    auto a = out.mutable_unchecked<3>();
    for (int e = 0; e < num_envs(); ++e)
      for (int i = 0; i < num_bodies(); ++i) {
        const auto &b = envs[e].bodies[i];
        double row[] = {b.x.x, b.x.y, b.x.z, b.q.w, b.q.x, b.q.y, b.q.z,
                        b.v.x, b.v.y, b.v.z, b.w.x, b.w.y, b.w.z};
        for (int k = 0; k < 13; ++k)
          a(e, i, k) = row[k];
      }
    return out;
  }
  Array joint_state() const {
    auto guard = lock();
    Array out({num_envs(), num_joints(), 2});
    auto a = out.mutable_unchecked<3>();
    for (int e = 0; e < num_envs(); ++e)
      for (int i = 0; i < num_joints(); ++i) {
        const auto &s = envs[e];
        const auto &j = s.joints[i];
        a(e, i, 0) = s.angle(j);
        a(e, i, 1) = duck::dot(s.bodies[j.b].w - s.bodies[j.a].w,
                               duck::rotate(s.bodies[j.a].q, j.axisA));
      }
    return out;
  }
  Array diagnostics() const {
    auto guard = lock();
    Array out({num_envs(), 5});
    auto a = out.mutable_unchecked<2>();
    for (int e = 0; e < num_envs(); ++e) {
      auto &d = controlDiagnostics[e];
      a(e, 0) = d.jointError;
      a(e, 1) = d.penetration;
      a(e, 2) = d.contacts;
      a(e, 3) = d.converged;
      a(e, 4) = d.axisError;
    }
    return out;
  }
};
PYBIND11_MODULE(_duck_cpu, m) {
  m.doc() = "Native CPU AVBD batch; no MuJoCo physics dependency";
  py::class_<CpuBatch>(m, "CpuBatch")
      .def(py::init<const std::string &, int, double, int, int>(),
           py::arg("model"), py::arg("num_envs") = 1, py::arg("dt") = .001,
           py::arg("iterations") = 200, py::arg("threads") = 1)
      .def_property_readonly("num_envs", &CpuBatch::num_envs)
      .def_property_readonly("joint_names", &CpuBatch::joint_names)
      .def("reset", &CpuBatch::reset)
      .def("step", &CpuBatch::step)
      .def("body_state", &CpuBatch::body_state)
      .def("joint_state", &CpuBatch::joint_state)
      .def("diagnostics", &CpuBatch::diagnostics);
}
