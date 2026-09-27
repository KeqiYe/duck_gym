#pragma once
#include "duck/solver.hpp"
#include <type_traits>
namespace duck_native {
inline void require(bool condition, const char *message) {
  if (!condition)
    throw std::invalid_argument(message);
}
template <class V> void setVec(V &dst, duck::Vec3 v) {
  dst.x = v.x;
  dst.y = v.y;
  dst.z = v.z;
}
template <class Q> void setQuat(Q &dst, duck::Quat q) {
  dst.w = q.w;
  dst.x = q.x;
  dst.y = q.y;
  dst.z = q.z;
}
template <class S>
void convert(const duck::Solver &original, typename S::ModelType &hostModel,
             std::vector<typename S::Point> &hostPoints, double dt,
             int iterations) {
  using Real = typename S::Scalar;
  using Point = typename S::Point;
  require(original.bodies.size() <= 32 && original.joints.size() <= 32 &&
              original.shapes.size() <= 32 &&
              original.collisionPairs.size() <= 32,
          "CUDA model capacity exceeded (32 bodies/joints/shapes)");
  require(original.bodies.size() > 1, "CUDA batch needs at least one body");
  require(original.bodies.size() <= S::bodyCapacity &&
              original.joints.size() <= S::jointCapacity,
          "Model exceeds selected state body/joint capacity");
  size_t maximumContacts = original.collisionPairs.size();
  if (original.options.ground)
    for (const auto &shape : original.shapes)
      if (shape.ground && original.bodies[shape.body].mass > 0)
        maximumContacts += 4;
  require(maximumContacts <= S::contactCapacity,
          "Model exceeds selected state contact capacity");
  hostModel.bodies.length = original.bodies.size();
  hostModel.joints.length = original.joints.size();
  hostModel.shapes.length = original.shapes.size();
  for (int i = 0; i < hostModel.bodies.size(); ++i) {
    auto &d = hostModel.bodies[i];
    auto &b = original.bodies[i];
    d.mass = b.mass;
    setVec(d.inertia, b.inertia);
    setVec(d.x, b.x);
    setVec(d.v, b.v);
    setVec(d.w, b.w);
    setQuat(d.q, b.q);
  }
  std::vector<int> incoming(original.bodies.size(), 0);
  for (auto &j : original.joints)
    require(++incoming[j.b] == 1, "A body must have at most one parent hinge");
  std::vector<bool> seen(original.bodies.size());
  for (int i = 0; i < int(seen.size()); ++i)
    seen[i] = incoming[i] == 0;
  for (int i = 0; i < hostModel.joints.size(); ++i) {
    auto &d = hostModel.joints[i];
    auto &j = original.joints[i];
    require(j.a < j.b, "CUDA reset requires topological body ordering");
    require(seen[j.a], "Joint list must follow parent-before-child ordering");
    seen[j.b] = true;
    d.a = j.a;
    d.b = j.b;
    hostModel.colors[j.b] = 1 - hostModel.colors[j.a];
    setVec(d.ra, j.ra);
    setVec(d.rb, j.rb);
    setVec(d.axisA, j.axisA);
    setVec(d.axisB, j.axisB);
    setVec(d.tangentA, j.tangentA);
    setVec(d.tangentB, j.tangentB);
    d.lo = j.lo;
    d.hi = j.hi;
    d.limited = j.limited;
    d.damping = j.damping;
    d.armature = j.armature;
    d.frictionloss = j.frictionloss;
    d.torque = j.torque;
    d.kp = j.kp;
    d.target = j.target;
    d.maxTorque = j.maxTorque;
    for (int k = 0; k < 7; ++k)
      d.penalty[k] = j.penalty[k];
  }
  for (auto &j : original.joints)
    require(original.bodies[j.a].mass == 0 || original.bodies[j.b].mass == 0 ||
                hostModel.colors[j.a] != hostModel.colors[j.b],
            "Joint graph must be bipartite");
  int adjacencySize = 0;
  for (int bi = 0; bi < hostModel.bodies.size(); ++bi) {
    hostModel.bodyJointOffsets[bi] = adjacencySize;
    for (int ji = 0; ji < hostModel.joints.size(); ++ji) {
      const auto &j = hostModel.joints[ji];
      if (j.a == bi || j.b == bi)
        hostModel.bodyJointIndices[adjacencySize++] = ji;
    }
  }
  hostModel.bodyJointOffsets[hostModel.bodies.size()] = adjacencySize;
  for (int i = 0; i < hostModel.shapes.size(); ++i) {
    auto &d = hostModel.shapes[i];
    auto &s = original.shapes[i];
    d.body = s.body;
    d.ground = s.ground;
    d.type = s.type;
    setVec(d.center, s.center);
    setVec(d.size, s.size);
    setQuat(d.q, s.q);
    d.friction = s.friction;
    d.boundingRadius = s.boundingRadius;
    d.offset = hostPoints.size();
    d.count = s.type == 0 ? 1 : (s.type == 2 ? 2 : s.localPoints.size());
    duck::Vec3 interior = s.center;
    if (!s.localPoints.empty()) {
      interior = {};
      for (auto p : s.localPoints)
        interior += p;
      interior = interior / double(s.localPoints.size());
    }
    setVec(d.interior, interior);
    for (auto p : s.localPoints) {
      Point v;
      setVec(v, p);
      hostPoints.push_back(v);
    }
  }
  hostModel.pairs.length = original.collisionPairs.size();
  for (int i = 0; i < hostModel.pairs.size(); ++i) {
    auto &d = hostModel.pairs[i];
    auto &p = original.collisionPairs[i];
    d.a = p.a;
    d.b = p.b;
    d.friction = p.friction;
  }
  hostModel.options.dt = dt;
  hostModel.options.iterations = iterations;
  setVec(hostModel.options.gravity, original.options.gravity);
  hostModel.options.ground = original.options.ground;
  // FP32 has an attainable stopping tolerance; parity tests can use FP64.
  hostModel.options.tolerance = std::is_same_v<Real, float> ? 1e-6 : 1e-7;
}
} // namespace duck_native
