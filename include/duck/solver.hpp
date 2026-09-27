#pragma once
#include "math.hpp"
#include <string>
#include <vector>
namespace duck {
struct Body {
  std::string name;
  double mass = 0;
  Vec3 inertia{}, x{}, v{}, w{}, x0{}, xp{}, force{};
  Quat q{}, q0{}, qp{};
};
struct Joint {
  std::string name;
  int a = 0, b = 0;
  Vec3 ra{}, rb{}, axisA{}, axisB{}, tangentA{}, tangentB{};
  double lo = 0, hi = 0, damping = 0, armature = 0, frictionloss = 0,
         lambdaF = 0, penaltyF = 1, torque = 0, kp = 0, target = 0,
         maxTorque = 0;
  bool limited = false;
  double angle0 = 0, velocity0 = 0;
  std::array<double, 7> lambda{}, penalty{}, c0{};
};
struct Shape {
  int body = 0;
  Vec3 center{}, size{};
  Quat q{};
  int type = 0;
  double friction = 0.8;
  std::vector<Vec3> vertices, localPoints;
  double boundingRadius = 0;
  bool ground = true;
};
struct CollisionPair {
  int a = 0, b = 0; // shape indices
  double friction = 0;
};
struct Contact {
  int body = 0, shape = 0, feature = 0;
  Vec3 local{}, anchor{};
  double radius = 0, friction = 0, c0 = 0;
  std::array<double, 3> lambda{}, penalty{};
  bool stick = false;
};
struct Options {
  double dt = 0.001, alpha = 0.9, gamma = 0.99, betaLinear = 1e7,
         betaAngular = 1e5, tolerance = 1e-7;
  int iterations = 100;
  Vec3 gravity{0, 0, -9.81};
  bool ground = false;
};
struct Diagnostics {
  double jointError = 0, axisError = 0, penetration = 0, update = 0;
  int contacts = 0, iterations = 0;
  bool converged = false;
};
class Solver {
public:
  Options options;
  std::vector<Body> bodies;
  std::vector<Joint> joints;
  std::vector<Shape> shapes;
  std::vector<CollisionPair> collisionPairs;
  std::vector<Contact> contacts;
  Diagnostics diagnostics;
  void load(const std::string &path);
  void step();
  void reset();
  double angle(const Joint &j) const;
  void evaluateJoint(const Joint &j, std::array<double, 7> &C,
                     std::array<V6, 7> &JA, std::array<V6, 7> &JB) const;

private:
  std::vector<std::vector<int>> adjacency;
  std::vector<Body> initialBodies;
  std::vector<Joint> initialJoints;
  void detectContacts();
  void contactValues(const Contact &c, Vec3 &C, std::array<V6, 3> &J) const;
};
// Each entry owns mutable state; CPU environments currently execute
// sequentially.
class Batch {
public:
  std::vector<Solver> environments;
  Batch(const Solver &prototype, int num_envs) {
    if (num_envs < 1)
      throw std::invalid_argument("num_envs must be positive");
    environments.assign(num_envs, prototype);
  }
  void step() {
    for (auto &env : environments)
      env.step();
  }
  void reset(int id) { environments.at(id).reset(); }
};
} // namespace duck
