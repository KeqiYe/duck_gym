#include "duck/solver.hpp"
#include <iostream>
using namespace duck;
void require(bool ok) {
  if (!ok)
    throw std::runtime_error("Math invariant failed");
}
int main() {
  for (Vec3 v : {Vec3{0, 0, 0}, Vec3{1e-10, -2e-10, 3e-10},
                 Vec3{0.4, -0.2, 1.3}, Vec3{3.1, 0, 0}}) {
    require(norm(log(exp(v)) - v) < 1e-12);
    Vec3 a{1, 2, 3};
    require(norm(rotate(conjugate(exp(v)), rotate(exp(v), a)) - a) < 1e-12);
  }
  M6 A{};
  V6 expected{1, -2, 3, -4, 5, -6}, g{};
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 6; ++j)
      A[i][j] = (i == j ? 10 : 0) + 1.0 / (i + j + 1);
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 6; ++j)
      g[i] -= A[i][j] * expected[j];
  auto x = solve(A, g);
  for (int i = 0; i < 6; ++i)
    require(std::abs(x[i] - expected[i]) < 1e-12);
  bool rejected = false;
  try {
    solve(M6{}, V6{});
  } catch (const std::exception &) {
    rejected = true;
  }
  require(rejected);
  Solver sim;
  sim.bodies.resize(2);
  sim.bodies[0].x = {.1, .2, .3};
  sim.bodies[1].x = {-.2, .3, .4};
  sim.bodies[0].q = exp({.2, -.3, .1});
  sim.bodies[1].q = exp({-.4, .1, .2});
  Joint joint;
  joint.a = 0;
  joint.b = 1;
  joint.ra = {.03, -.02, .07};
  joint.rb = {-.01, .05, .03};
  joint.axisA = {0, 0, 1};
  joint.axisB = {0, 1, 0};
  joint.tangentA = {1, 0, 0};
  joint.tangentB = {1, 0, 0};
  std::array<double, 7> C{};
  std::array<V6, 7> JA{}, JB{};
  sim.evaluateJoint(joint, C, JA, JB);
  for (int body = 0; body < 2; ++body)
    for (int dof = 0; dof < 6; ++dof) {
      Body saved = sim.bodies[body];
      std::array<double, 7> plus{}, minus{};
      std::array<V6, 7> scratchA{}, scratchB{};
      Vec3 delta{};
      delta[dof % 3] = 1e-6;
      if (dof < 3)
        sim.bodies[body].x = saved.x + delta;
      else
        sim.bodies[body].q = exp(delta) * saved.q;
      sim.evaluateJoint(joint, plus, scratchA, scratchB);
      sim.bodies[body] = saved;
      if (dof < 3)
        sim.bodies[body].x = saved.x - delta;
      else
        sim.bodies[body].q = exp(-delta) * saved.q;
      sim.evaluateJoint(joint, minus, scratchA, scratchB);
      sim.bodies[body] = saved;
      for (int row = 0; row < 5; ++row)
        require(std::abs((plus[row] - minus[row]) / 2e-6 -
                         (body == 0 ? JA : JB)[row][dof]) < 1e-8);
    }
  std::cout
      << "SO(3), quaternion, SPD solve and hinge Jacobian invariants passed\n";
}
