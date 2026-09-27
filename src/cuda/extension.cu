#include "duck/solver.hpp"
#include "model.hpp"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstring>
#include <cuda_runtime.h>
#include <memory>
#include <torch/extension.h>
#include <type_traits>
#define DUCK_NAMESPACE gpu32
#define DUCK_SCALAR float
#include "kernel.cuh"
#undef DUCK_NAMESPACE
#undef DUCK_SCALAR
#define DUCK_NAMESPACE gpu64
#define DUCK_SCALAR double
#include "kernel.cuh"
#undef DUCK_NAMESPACE
#undef DUCK_SCALAR

template <class S>
__global__ void initialize(S *states, const typename S::ModelType *model,
                           const typename S::Point *points, int count) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e < count)
    states[e].initialize(model, points);
}
template <class S>
__global__ void resetStates(S *states, const typename S::ModelType *model,
                            const typename S::Point *points, int count,
                            const bool *mask, const typename S::Scalar *q,
                            const typename S::Scalar *roots) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= count || !mask[e])
    return;
  for (int k = 0; k < 6; ++k)
    if (!isfinite(roots[e * 6 + k])) {
      states[e].failed = 1;
      return;
    }
  for (int k = 0; k < model->joints.size(); ++k)
    if (!isfinite(q[e * model->joints.size() + k])) {
      states[e].failed = 1;
      return;
    }
  auto &s = states[e];
  s.initialize(model, points);
  s.resetPose(q + e * model->joints.size(), roots + e * 6);
}
// Compact MicroDuck states can stay on chip for all AVBD iterations/substeps.
// Copy cooperatively; solver arithmetic and its per-environment order are unchanged.
template <class S, bool Shared>
__device__ S &loadWorkingState(S *states, int e, unsigned char *storage) {
  if constexpr (Shared) {
    static_assert(std::is_trivially_copyable_v<S>);
    auto *dst = reinterpret_cast<unsigned char *>(storage);
    auto *src = reinterpret_cast<const unsigned char *>(states + e);
    for (int i = threadIdx.x; i < sizeof(S); i += blockDim.x)
      dst[i] = src[i];
    __syncthreads();
    return *reinterpret_cast<S *>(storage);
  } else {
    return states[e];
  }
}
template <class S, bool Shared>
__device__ void storeWorkingState(S *states, int e, unsigned char *storage) {
  if constexpr (Shared) {
    __syncthreads();
    auto *dst = reinterpret_cast<unsigned char *>(states + e);
    auto *src = reinterpret_cast<const unsigned char *>(storage);
    for (int i = threadIdx.x; i < sizeof(S); i += blockDim.x)
      dst[i] = src[i];
  }
}

template <class S, bool Shared = false>
__global__ void advance(S *states, int count, const typename S::Scalar *targets,
                        const typename S::Scalar *forces, int substeps,
                        typename S::Scalar kp, typename S::Scalar limit) {
  int e = blockIdx.x;
  if (e >= count)
    return;
  extern __shared__ __align__(16) unsigned char workspace[];
  auto &s = loadWorkingState<S, Shared>(states, e, workspace);
  if (threadIdx.x == 0) {
    for (int j = 0; j < s.joints.size(); ++j)
      if (!isfinite(targets[e * s.joints.size() + j]))
        s.failed = 1;
    for (int k = 0; k < 3; ++k)
      if (!isfinite(forces[e * 3 + k]))
        s.failed = 1;
  }
  __syncthreads();
  if (s.failed) {
    storeWorkingState<S, Shared>(states, e, workspace);
    return;
  }
  if (threadIdx.x == 0) {
    for (int j = 0; j < s.joints.size(); ++j) {
      s.joints[j].target = targets[e * s.joints.size() + j];
      s.joints[j].kp = kp;
      s.joints[j].maxTorque = limit;
      s.joints[j].torque = s.model->joints[j].torque;
      s.joints[j].frictionloss = s.model->joints[j].frictionloss;
      s.joints[j].damping = s.model->joints[j].damping;
    }
    s.bodies[1].force = {forces[3 * e], forces[3 * e + 1], forces[3 * e + 2]};
  }
  __syncthreads();
  s.advance(substeps);
  storeWorkingState<S, Shared>(states, e, workspace);
}
template <class S, bool Shared = false>
__global__ void advanceMotor(S *states, const typename S::Scalar *motor,
                             const typename S::Scalar *forces) {
  int e = blockIdx.x;
  extern __shared__ __align__(16) unsigned char workspace[];
  auto &s = loadWorkingState<S, Shared>(states, e, workspace);
  if (threadIdx.x == 0)
    s.setMotor(motor + e * s.joints.size() * 3, forces + e * 3);
  __syncthreads();
  if (!s.failed)
    s.advance(1);
  storeWorkingState<S, Shared>(states, e, workspace);
}
template <class S>
__global__ void snapshot(const S *states, int count, typename S::Scalar *bodies,
                         typename S::Scalar *joints,
                         typename S::Scalar *diagnostics) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= count)
    return;
  const auto &s = states[e];
  s.snapshot(bodies + e * s.bodies.size() * 13,
             joints + e * s.joints.size() * 2, diagnostics + e * 6);
}
template <class S>
__global__ void groundForces(const S *states, int count,
                             typename S::Scalar *out, bool groundOnly) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e < count)
    states[e].contactForces(out + e * states[e].bodies.size() * 3, groundOnly);
}
template <class S>
__global__ void selfContactStatsKernel(const S *states, int count,
                                       typename S::Scalar *out) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e < count)
    states[e].selfContactStats(out + 2 * e);
}
template <class S>
__global__ void contactCountsKernel(const S *states, int count,
                                     typename S::Scalar *out) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e < count)
    states[e].contactCounts(out + 2 * e);
}
struct Interface {
  virtual ~Interface() = default;
  virtual void reset(torch::Tensor, torch::Tensor, torch::Tensor) = 0;
  virtual void step(torch::Tensor, torch::Tensor, int, double, double) = 0;
  virtual void stepMotor(torch::Tensor, torch::Tensor) = 0;
  virtual std::vector<torch::Tensor> state() = 0;
  virtual torch::Tensor contactForces(bool) = 0;
  virtual torch::Tensor selfContactStats() = 0;
  virtual torch::Tensor contactCounts() = 0;
  virtual torch::Tensor generalizedLoads(torch::Tensor) = 0;
  virtual std::vector<torch::Tensor> reference(torch::Tensor, torch::Tensor,
                                               int, double, double) = 0;
  virtual std::vector<std::string> names() = 0;
};

template <class S>
__global__ void generalizedLoadsKernel(const S *states,
                                       const typename S::Scalar *offset,
                                       typename S::Scalar *out, int count) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e < count) {
    auto p = offset + 3 * e;
    states[e].generalizedLoads({p[0], p[1], p[2]},
                               out + 3 * e * states[e].joints.size());
  }
}

template <class S> class Batch final : public Interface {
  static constexpr bool compact = S::bodyCapacity == 16;
  bool sharedMemory;
  using Real = typename S::Scalar;
  using Model = typename S::ModelType;
  using Point = typename S::Point;
  duck::Solver original;
  Model hostModel{};
  std::vector<Point> hostPoints;
  torch::Tensor modelStorage, pointStorage, stateStorage;
  cudaEvent_t ready = nullptr;
  int count, device;
  torch::ScalarType dtype;
  Model *model() { return reinterpret_cast<Model *>(modelStorage.data_ptr()); }
  Point *points() { return reinterpret_cast<Point *>(pointStorage.data_ptr()); }
  S *states() { return reinterpret_cast<S *>(stateStorage.data_ptr()); }
  cudaStream_t stream() { return c10::cuda::getCurrentCUDAStream(device); }
  void wait() { C10_CUDA_CHECK(cudaStreamWaitEvent(stream(), ready, 0)); }
  void record() { C10_CUDA_CHECK(cudaEventRecord(ready, stream())); }
  void check(torch::Tensor t, std::vector<int64_t> shape,
             torch::ScalarType type) {
    TORCH_CHECK(t.is_cuda() && t.get_device() == device &&
                    t.scalar_type() == type && t.is_contiguous() &&
                    t.sizes().vec() == shape,
                "Invalid CUDA tensor device/dtype/shape/contiguity");
    t.record_stream(c10::cuda::getCurrentCUDAStream(device).unwrap());
  }

public:
  Batch(std::string path, int n, double dt, int iterations, int gpu, bool shared = false)
      : sharedMemory(shared), count(n), device(gpu),
        dtype(std::is_same_v<Real, float> ? torch::kFloat32 : torch::kFloat64) {
    TORCH_CHECK(n > 0 && n <= 65536 && dt > 0 && std::isfinite(dt) &&
                    iterations > 0,
                "Invalid batch parameters");
    c10::cuda::CUDAGuard guard(device);
    original.load(path);
    duck_native::convert<S>(original, hostModel, hostPoints, dt, iterations);
    if constexpr (compact) {
      TORCH_CHECK(original.bodies.size() == 16 && original.joints.size() == 14,
                  "MicroDuck specialized backend requires 16 bodies and 14 hinges");
      const char *expected[] = {
          "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
          "neck_pitch", "head_pitch", "head_yaw", "head_roll",
          "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"};
      for (int i = 0; i < 14; ++i) {
        const auto &label = original.joints[i].name;
        const auto name = label.rfind("robot/", 0) == 0 ? label.substr(6) : label;
        TORCH_CHECK(name == expected[i], "Unsupported MicroDuck joint layout");
      }
    }
    if (sharedMemory) {
      int maxShared = 0;
      C10_CUDA_CHECK(cudaDeviceGetAttribute(&maxShared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
      TORCH_CHECK(sizeof(S) <= size_t(maxShared), "Device shared memory is too small for MicroDuck state");
      C10_CUDA_CHECK(cudaFuncSetAttribute(advance<S, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(S)));
      C10_CUDA_CHECK(cudaFuncSetAttribute(advanceMotor<S, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(S)));
    }
    auto byteOptions = torch::TensorOptions()
                           .dtype(torch::kUInt8)
                           .device(torch::kCUDA, device);
    modelStorage = torch::empty({int64_t(sizeof(Model))}, byteOptions);
    pointStorage = torch::empty(
        {int64_t(std::max<size_t>(1, hostPoints.size()) * sizeof(Point))},
        byteOptions);
    stateStorage = torch::zeros({int64_t(count * sizeof(S))}, byteOptions);
    C10_CUDA_CHECK(cudaMemcpyAsync(model(), &hostModel, sizeof(Model),
                                   cudaMemcpyHostToDevice, stream()));
    if (!hostPoints.empty())
      C10_CUDA_CHECK(cudaMemcpyAsync(points(), hostPoints.data(),
                                     hostPoints.size() * sizeof(Point),
                                     cudaMemcpyHostToDevice, stream()));
    initialize<<<(n + 31) / 32, 32, 0, stream()>>>(states(), model(), points(),
                                                   n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    C10_CUDA_CHECK(cudaStreamSynchronize(stream()));
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    record();
  }
  ~Batch() {
    c10::cuda::CUDAGuard guard(device);
    if (ready) {
      cudaEventSynchronize(ready);
      cudaEventDestroy(ready);
    }
  }
  void reset(torch::Tensor mask, torch::Tensor q, torch::Tensor root) override {
    c10::cuda::CUDAGuard guard(device);
    check(mask, {count}, torch::kBool);
    check(q, {count, hostModel.joints.size()}, dtype);
    check(root, {count, 6}, dtype);
    wait();
    resetStates<<<(count + 31) / 32, 32, 0, stream()>>>(
        states(), model(), points(), count, mask.data_ptr<bool>(),
        q.template data_ptr<Real>(), root.template data_ptr<Real>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    record();
  }
  void step(torch::Tensor target, torch::Tensor forces, int substeps, double kp,
            double limit) override {
    c10::cuda::CUDAGuard guard(device);
    check(target, {count, hostModel.joints.size()}, dtype);
    check(forces, {count, 3}, dtype);
    // Zero selects the solver's uncapped PD spring for the explicit common
    // physics comparison. Negative and nonfinite caps remain invalid.
    TORCH_CHECK(substeps > 0 && kp >= 0 && std::isfinite(kp) && limit >= 0 &&
                    std::isfinite(limit),
                "Invalid controls");
    wait();
    if (sharedMemory)
      advance<S, true><<<count, 32, sizeof(S), stream()>>>(
          states(), count, target.template data_ptr<Real>(),
          forces.template data_ptr<Real>(), substeps, Real(kp), Real(limit));
    else
      advance<S, false><<<count, 32, 0, stream()>>>(
          states(), count, target.template data_ptr<Real>(),
          forces.template data_ptr<Real>(), substeps, Real(kp), Real(limit));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    record();
  }
  void stepMotor(torch::Tensor motor, torch::Tensor forces) override {
    c10::cuda::CUDAGuard guard(device);
    check(motor, {count, hostModel.joints.size(), 3}, dtype);
    check(forces, {count, 3}, dtype);
    wait();
    if (sharedMemory)
      advanceMotor<S, true><<<count, 32, sizeof(S), stream()>>>(states(),
          motor.template data_ptr<Real>(), forces.template data_ptr<Real>());
    else
      advanceMotor<S, false><<<count, 32, 0, stream()>>>(states(),
          motor.template data_ptr<Real>(), forces.template data_ptr<Real>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    record();
  }
  std::vector<torch::Tensor> state() override {
    c10::cuda::CUDAGuard guard(device);
    auto options =
        torch::TensorOptions().dtype(dtype).device(torch::kCUDA, device);
    auto b = torch::empty({count, hostModel.bodies.size(), 13}, options),
         j = torch::empty({count, hostModel.joints.size(), 2}, options),
         d = torch::empty({count, 6}, options);
    wait();
    snapshot<<<(count + 31) / 32, 32, 0, stream()>>>(
        states(), count, b.template data_ptr<Real>(),
        j.template data_ptr<Real>(), d.template data_ptr<Real>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    record();
    return {b, j, d};
  }
  torch::Tensor contactForces(bool groundOnly) override {
    c10::cuda::CUDAGuard guard(device);
    auto out = torch::empty(
        {count, hostModel.bodies.size(), 3},
        torch::TensorOptions().dtype(dtype).device(torch::kCUDA, device));
    wait();
    groundForces<<<(count + 31) / 32, 32, 0, stream()>>>(
        states(), count, out.template data_ptr<Real>(), groundOnly);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    record();
    return out;
  }
  torch::Tensor selfContactStats() override {
    c10::cuda::CUDAGuard guard(device);
    auto out = torch::empty(
        {count, 2},
        torch::TensorOptions().dtype(dtype).device(torch::kCUDA, device));
    wait();
    selfContactStatsKernel<<<(count + 31) / 32, 32, 0, stream()>>>(
        states(), count, out.template data_ptr<Real>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    record();
    return out;
  }
  torch::Tensor contactCounts() override {
    c10::cuda::CUDAGuard guard(device);
    auto out = torch::empty({count, 2},
        torch::TensorOptions().dtype(dtype).device(torch::kCUDA, device));
    wait();
    contactCountsKernel<<<(count + 31) / 32, 32, 0, stream()>>>(
        states(), count, out.template data_ptr<Real>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    record();
    return out;
  }
  torch::Tensor generalizedLoads(torch::Tensor offset) override {
    c10::cuda::CUDAGuard guard(device);
    check(offset, {count, 3}, dtype);
    auto out = torch::empty(
        {count, hostModel.joints.size(), 3},
        torch::TensorOptions().dtype(dtype).device(torch::kCUDA, device));
    wait();
    generalizedLoadsKernel<<<(count + 31) / 32, 32, 0, stream()>>>(
        states(), offset.template data_ptr<Real>(),
        out.template data_ptr<Real>(), count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    record();
    return out;
  }
  std::vector<torch::Tensor> reference(torch::Tensor target,
                                       torch::Tensor forces, int substeps,
                                       double kp, double limit) override {
    TORCH_CHECK(target.device().is_cpu() && target.is_contiguous() &&
                    target.scalar_type() == dtype &&
                    target.sizes().vec() ==
                        std::vector<int64_t>({hostModel.joints.size()}),
                "Invalid CPU reference target");
    TORCH_CHECK(forces.device().is_cpu() && forces.is_contiguous() &&
                    forces.sizes().vec() == std::vector<int64_t>({3}) &&
                    forces.scalar_type() == dtype && substeps > 0,
                "Invalid CPU reference inputs");
    S s;
    s.initialize(&hostModel, hostPoints.data());
    auto options = torch::TensorOptions().dtype(dtype);
    auto b = torch::empty({hostModel.bodies.size(), 13}, options),
         j = torch::empty({hostModel.joints.size(), 2}, options),
         d = torch::empty({6}, options);
    s.bodies[1].force = {forces.template data_ptr<Real>()[0],
                         forces.template data_ptr<Real>()[1],
                         forces.template data_ptr<Real>()[2]};
    for (int i = 0; i < s.joints.size(); ++i) {
      s.joints[i].kp = kp;
      s.joints[i].maxTorque = limit;
      s.joints[i].target = target.template data_ptr<Real>()[i];
    }
    s.advance(substeps);
    s.snapshot(b.template data_ptr<Real>(), j.template data_ptr<Real>(),
               d.template data_ptr<Real>());
    return {b, j, d};
  }
  std::vector<std::string> names() override {
    std::vector<std::string> n;
    for (auto &j : original.joints)
      n.push_back(j.name);
    return n;
  }
};
std::shared_ptr<Interface> makeBatch(std::string path, int n, double dt,
                                     int iterations, int device, bool fp64, bool specialized, bool sharedMemory) {
  TORCH_CHECK(specialized || !sharedMemory, "Shared memory requires the compact MicroDuck backend");
  if (specialized) {
    if (fp64)
      return std::make_shared<Batch<gpu64::MicroDuckState>>(path, n, dt, iterations, device, sharedMemory);
    return std::make_shared<Batch<gpu32::MicroDuckState>>(path, n, dt, iterations, device, sharedMemory);
  }
  if (fp64)
    return std::make_shared<Batch<gpu64::State>>(path, n, dt, iterations,
                                                 device);
  return std::make_shared<Batch<gpu32::State>>(path, n, dt, iterations, device);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<Interface, std::shared_ptr<Interface>>(m, "CudaBatch")
      .def(pybind11::init(&makeBatch), pybind11::arg("model"),
           pybind11::arg("num_envs"), pybind11::arg("dt") = .001,
           pybind11::arg("iterations") = 200, pybind11::arg("device") = 0,
           pybind11::arg("fp64") = false, pybind11::arg("specialized") = false, pybind11::arg("shared_memory") = false)
      .def("reset", &Interface::reset)
      .def("step", &Interface::step)
      .def("step_motor", &Interface::stepMotor)
      .def("state", &Interface::state)
      .def("contact_forces", &Interface::contactForces,
           pybind11::arg("ground_only") = false)
      .def("self_contact_stats", &Interface::selfContactStats)
      .def("contact_counts", &Interface::contactCounts)
      .def("generalized_loads", &Interface::generalizedLoads)
      .def("reference", &Interface::reference)
      .def_property_readonly("joint_names", &Interface::names);
}
