#include "duck/solver.hpp"
#include <iostream>
using namespace duck;
void require(bool b) {
  if (!b)
    throw std::runtime_error("State invariant failed");
}
int main(int argc, char **argv) {
  require(argc == 2);
  Solver original;
  original.load(argv[1]);
  original.options.iterations = 100;
  Batch batch(original, 3);
  batch.environments[1].bodies[1].v.x = 0.1;
  for (int i = 0; i < 50; ++i)
    batch.step();
  require(norm(batch.environments[0].bodies[1].x -
               batch.environments[2].bodies[1].x) < 1e-14);
  require(norm(batch.environments[0].bodies[1].x -
               batch.environments[1].bodies[1].x) > 1e-7);
  auto reference = batch.environments[2].bodies[1];
  batch.reset(0);
  require(norm(batch.environments[0].bodies[1].x - original.bodies[1].x) <
          1e-14);
  require(norm(batch.environments[2].bodies[1].x - reference.x) < 1e-14);
  for (int i = 0; i < 50; ++i)
    batch.environments[0].step();
  require(norm(batch.environments[0].bodies[1].x - reference.x) < 1e-14);
  batch.environments[0].load(argv[1]);
  for (auto &j : batch.environments[0].joints)
    for (double l : j.lambda)
      require(l == 0);
  bool rejected = false;
  try {
    Batch invalid(original, 0);
  } catch (const std::exception &) {
    rejected = true;
  }
  require(rejected);
  std::cout << "Reset, repeated load and independent environments passed\n";
}
