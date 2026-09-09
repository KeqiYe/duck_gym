/* AVBD warm-start and penalty scheduling follow the author reference:
 * https://github.com/savant117/avbd-demo3d/tree/7701bd427d55ca5d03ea1fdf331912ded9169f4b
 * Copyright (c) 2026 Chris Giles
 * Permission to use, copy, modify, distribute and sell this software
 * and its documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies.
 * Chris Giles makes no representations about the suitability
 * of this software for any purpose.
 * It is provided "as is" without express or implied warranty.
 *
 * duck_gym uses its own FP64 math, hinge representation, SO(3) integration,
 * parser and ground contact implementation; it is not the reference executable.
 */
#include "duck/solver.hpp"
#include <fstream>
#include <limits>
#include <map>
#include <tuple>

namespace duck {
namespace {
Vec3 readV(std::istream &in) {
  Vec3 v;
  in >> v.x >> v.y >> v.z;
  return v;
}
Quat readQ(std::istream &in) {
  Quat q;
  in >> q.w >> q.x >> q.y >> q.z;
  return normalized(q);
}
Vec3 inertia(const Body &b, Vec3 w, bool inverse = false) {
  Vec3 local = rotate(conjugate(b.q0), w);
  for (int k = 0; k < 3; ++k)
    local[k] *= inverse ? 1 / b.inertia[k] : b.inertia[k];
  return rotate(b.q0, local);
}
// Positive diagonal majorization of rotational geometric stiffness (AVBD
// Sec. 3.5).
void geometric(M6 &H, Vec3 r, Vec3 f) {
  double rf = dot(r, f);
  for (int col = 0; col < 3; ++col) {
    double sum = 0;
    for (int row = 0; row < 3; ++row) {
      double a =
          0.5 * (r[row] * f[col] + f[row] * r[col]) - (row == col ? rf : 0);
      sum += a * a;
    }
    H[col + 3][col + 3] += std::sqrt(sum);
  }
}
double minimumPenalty(int k) { return k < 3 ? 1000 : 1; }
double maximumPenalty(int k) { return k < 3 ? 1e9 : 1e7; }
Vec3 contactForce(Vec3 C, const Contact &c) {
  Vec3 f;
  for (int k = 0; k < 3; ++k)
    f[k] = c.lambda[k] + c.penalty[k] * C[k];
  f.z = std::min(f.z, 0.0);
  double t = std::hypot(f.x, f.y), bound = -c.friction * f.z;
  if (t > bound && t > 0) {
    f.x *= bound / t;
    f.y *= bound / t;
  }
  return f;
}
} // namespace
void Solver::load(const std::string &path) {
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("Cannot read model: " + path);
  std::string header;
  int version, nbody, njoint, nshape;
  in >> header >> version >> nbody >> njoint >> nshape;
  if (header != "DUCK_MODEL" || version != 1 || nbody < 1 || nbody > 10000 ||
      njoint < 0 || nshape < 0)
    throw std::runtime_error("Invalid model header");
  options.gravity = readV(in);
  in >> options.ground;
  bodies.assign(nbody, Body{});
  joints.assign(njoint, Joint{});
  shapes.assign(nshape, Shape{});
  contacts.clear();
  adjacency.assign(nbody, {});
  for (auto &b : bodies) {
    in >> b.name >> b.mass;
    b.inertia = readV(in);
    b.x = readV(in);
    b.q = readQ(in);
    b.v = readV(in);
    b.w = readV(in);
    if (b.mass < 0 || (b.mass > 0 && (b.inertia.x <= 0 || b.inertia.y <= 0 ||
                                      b.inertia.z <= 0)))
      throw std::runtime_error("Invalid mass/inertia");
  }
  for (int id = 0; id < njoint; ++id) {
    auto &j = joints[id];
    in >> j.name >> j.a >> j.b;
    j.ra = readV(in);
    j.rb = readV(in);
    j.axisA = unit(readV(in));
    j.axisB = unit(readV(in));
    j.tangentA = unit(readV(in));
    j.tangentB = unit(readV(in));
    in >> j.limited >> j.lo >> j.hi >> j.damping >> j.armature >>
        j.frictionloss >> j.torque >> j.kp >> j.target >> j.maxTorque;
    if (j.a < 0 || j.b < 0 || j.a >= nbody || j.b >= nbody || j.a == j.b ||
        j.damping < 0 || j.armature < 0 || j.frictionloss < 0)
      throw std::runtime_error("Invalid joint");
    for (int k = 0; k < 7; ++k)
      j.penalty[k] = minimumPenalty(k);
    adjacency[j.a].push_back(id);
    adjacency[j.b].push_back(id);
  }
  for (auto &s : shapes) {
    int n;
    in >> s.body >> s.type;
    s.center = readV(in);
    s.q = readQ(in);
    s.size = readV(in);
    in >> s.friction >> n;
    if (s.body < 0 || s.body >= nbody || s.type < 0 || s.type > 3 || n < 0 ||
        n > 1000000 || s.friction < 0)
      throw std::runtime_error("Invalid collision shape");
    for (int k = 0; k < n; ++k)
      s.vertices.push_back(readV(in));
  }
  if (!in)
    throw std::runtime_error("Truncated model");
  std::string extra;
  if (in >> extra)
    throw std::runtime_error("Unexpected model data");
  initialBodies = bodies;
  initialJoints = joints;
  diagnostics = {};
}
void Solver::reset() {
  bodies = initialBodies;
  joints = initialJoints;
  contacts.clear();
  diagnostics = {};
}
double Solver::angle(const Joint &j) const {
  const auto &a = bodies[j.a];
  const auto &b = bodies[j.b];
  Vec3 axis = rotate(a.q, j.axisA), t = rotate(a.q, j.tangentA),
       u = rotate(b.q, j.tangentB);
  return std::atan2(dot(axis, cross(t, u)), dot(t, u));
}
void Solver::evaluateJoint(const Joint &j, std::array<double, 7> &C,
                           std::array<V6, 7> &JA, std::array<V6, 7> &JB) const {
  const auto &a = bodies[j.a];
  const auto &b = bodies[j.b];
  Vec3 ra = rotate(a.q, j.ra), rb = rotate(b.q, j.rb), d = a.x + ra - b.x - rb;
  for (int k = 0; k < 3; ++k) {
    Vec3 e{};
    e[k] = 1;
    C[k] = d[k];
    JA[k] = row(e, cross(ra, e));
    JB[k] = row(-e, -cross(rb, e));
  }
  Vec3 axis = rotate(a.q, j.axisA), axisB = rotate(b.q, j.axisB),
       t1 = rotate(a.q, j.tangentA), t2 = cross(axis, t1);
  for (int k = 3; k < 5; ++k) {
    Vec3 t = k == 3 ? t1 : t2;
    C[k] = dot(t, axisB);
    JA[k] = row({}, cross(t, axisB));
    JB[k] = row({}, cross(axisB, t));
  }
  double q = angle(j);
  C[5] = q - j.lo;
  C[6] = j.hi - q;
  JA[5] = row({}, -axis);
  JB[5] = row({}, axis);
  JA[6] = row({}, axis);
  JB[6] = row({}, -axis);
}
void Solver::detectContacts() {
  auto old = std::move(contacts);
  contacts.clear();
  if (!options.ground)
    return;
  std::map<std::pair<int, int>, Contact> cache;
  for (auto &c : old)
    cache[{c.shape, c.feature}] = c;
  for (int si = 0; si < (int)shapes.size(); ++si) {
    auto &s = shapes[si];
    auto &b = bodies[s.body];
    if (b.mass == 0)
      continue;
    std::vector<Vec3> points;
    if (s.type == 0) {
      Vec3 down = rotate(conjugate(b.q), {0, 0, -s.size.x});
      points.push_back(s.center + down);
    } else if (s.type == 1) {
      for (int i = 0; i < 8; ++i)
        points.push_back(s.center + rotate(s.q, {(i & 1 ? 1 : -1) * s.size.x,
                                                 (i & 2 ? 1 : -1) * s.size.y,
                                                 (i & 4 ? 1 : -1) * s.size.z}));
    } else if (s.type == 2) {
      Vec3 down = rotate(conjugate(b.q), {0, 0, -s.size.x});
      for (int i = 0; i < 2; ++i)
        points.push_back(s.center +
                         rotate(s.q, {0, 0, (i ? 1 : -1) * s.size.y}) + down);
    } else {
      for (auto v : s.vertices)
        points.push_back(s.center + rotate(s.q, v));
    }
    std::vector<std::pair<double, int>> candidates;
    for (int k = 0; k < (int)points.size(); ++k) {
      Vec3 world = b.x + rotate(b.q, points[k]);
      Vec3 velocity = b.v + cross(b.w, world - b.x);
      double z =
          std::min(world.z, world.z + options.dt * velocity.z +
                                options.dt * options.dt * options.gravity.z);
      if (z < 0.003)
        candidates.push_back({z, k});
    }
    std::sort(candidates.begin(), candidates.end());
    // Preserve a small spatially separated ground manifold, never a mesh
    // bounding box.
    std::vector<Vec3> chosen;
    for (auto [z, k] : candidates) {
      (void)z;
      Vec3 world = b.x + rotate(b.q, points[k]);
      bool duplicate = false;
      for (auto v : chosen)
        if (std::hypot(v.x - world.x, v.y - world.y) < 0.005)
          duplicate = true;
      if (duplicate)
        continue;
      Contact c;
      c.body = s.body;
      c.shape = si;
      c.feature = k;
      c.local = points[k];
      c.anchor = {world.x, world.y, 0};
      c.friction = s.friction;
      c.penalty = {1000, 1000, 1000};
      auto it = cache.find({si, k});
      if (it != cache.end()) {
        c.lambda = it->second.lambda;
        c.penalty = it->second.penalty;
        if (it->second.stick) {
          c.local = it->second.local;
          c.anchor = it->second.anchor;
        }
      }
      c.c0 = std::min((b.x + rotate(b.q, c.local)).z, 0.0);
      for (int d = 0; d < 3; ++d) {
        c.lambda[d] *= options.alpha * options.gamma;
        c.penalty[d] = std::clamp(c.penalty[d] * options.gamma, 1000.0, 1e9);
      }
      contacts.push_back(c);
      chosen.push_back(world);
      if (chosen.size() == 4)
        break;
    }
  }
}
void Solver::contactValues(const Contact &c, Vec3 &C,
                           std::array<V6, 3> &J) const {
  const auto &b = bodies[c.body];
  Vec3 r = rotate(b.q, c.local);
  C = b.x + r - c.anchor;
  C.z -= options.alpha * c.c0;
  for (int k = 0; k < 3; ++k) {
    Vec3 e{};
    e[k] = 1;
    J[k] = row(e, cross(r, e));
  }
}
void Solver::step() {
  const double h = options.dt;
  if (!(h > 0) || options.iterations < 1 || options.alpha < 0 ||
      options.alpha >= 1 || options.gamma <= 0 || options.gamma >= 1)
    throw std::runtime_error("Invalid solver options");
  diagnostics = {};
  detectContacts();
  for (auto &j : joints) {
    std::array<V6, 7> A{}, B{};
    evaluateJoint(j, j.c0, A, B);
    j.angle0 = angle(j);
    j.lambdaF *= options.alpha * options.gamma;
    j.penaltyF = std::clamp(j.penaltyF * options.gamma, 1.0, 1e7);
    Vec3 axis = rotate(bodies[j.a].q, j.axisA);
    j.velocity0 = dot(bodies[j.b].w - bodies[j.a].w, axis);
    for (int k = 0; k < 7; ++k) {
      j.lambda[k] *= options.alpha * options.gamma;
      j.penalty[k] = std::clamp(j.penalty[k] * options.gamma, minimumPenalty(k),
                                maximumPenalty(k));
    }
  }
  std::vector<Vec3> torques(bodies.size());
  for (auto &j : joints) {
    Vec3 axis = rotate(bodies[j.a].q, j.axisA);
    double t = j.torque + j.kp * wrap(j.target - j.angle0);
    if (j.maxTorque > 0)
      t = std::clamp(t, -j.maxTorque, j.maxTorque);
    torques[j.b] += axis * t;
    torques[j.a] += axis * (-t);
  }
  for (int i = 0; i < (int)bodies.size(); ++i) {
    auto &b = bodies[i];
    b.x0 = b.x;
    b.q0 = b.q;
    if (b.mass == 0)
      continue;
    b.xp = b.x + h * b.v + h * h * options.gravity;
    Vec3 wp =
        b.w + h * inertia(b, torques[i] - cross(b.w, inertia(b, b.w)), true);
    b.qp = exp(h * wp) * b.q;
    b.x = b.xp;
    b.q = b.qp;
  }
  for (int it = 0; it < options.iterations; ++it) {
    double maxUpdate = 0;
    for (int bi = 0; bi < (int)bodies.size(); ++bi) {
      auto &b = bodies[bi];
      if (b.mass == 0)
        continue;
      M6 H{};
      V6 g{};
      Vec3 dx = b.x - b.xp, theta = log(b.q * conjugate(b.qp));
      for (int k = 0; k < 3; ++k) {
        H[k][k] = b.mass / (h * h);
        g[k] = H[k][k] * dx[k];
      }
      // SO(3) log Jacobian under left perturbations, with frozen world inertia.
      double a = norm(theta),
             coef = a < 1e-5 ? 1.0 / 12 + a * a / 720
                             : (1 - 0.5 * a / std::tan(0.5 * a)) / (a * a);
      Vec3 rotJ[3];
      for (int k = 0; k < 3; ++k) {
        Vec3 e{};
        e[k] = 1;
        rotJ[k] =
            e - 0.5 * cross(theta, e) + coef * cross(theta, cross(theta, e));
      }
      for (int k = 0; k < 3; ++k) {
        g[k + 3] = dot(rotJ[k], inertia(b, theta)) / (h * h);
        for (int l = 0; l < 3; ++l)
          H[k + 3][l + 3] = dot(rotJ[k], inertia(b, rotJ[l])) / (h * h);
      }
      for (int ji : adjacency[bi]) {
        auto &j = joints[ji];
        std::array<double, 7> C{};
        std::array<V6, 7> A{}, B{};
        evaluateJoint(j, C, A, B);
        const auto &J = bi == j.a ? A : B;
        for (int k = 0; k < (j.limited ? 7 : 5); ++k) {
          double c =
              C[k] - options.alpha * (k < 5 ? j.c0[k] : std::min(j.c0[k], 0.0));
          double f = j.lambda[k] + j.penalty[k] * c;
          if (k >= 5)
            f = std::min(f, 0.0);
          if (k < 5 || f < 0) {
            stamp(H, g, J[k], f, j.penalty[k]);
            if (k < 3) {
              Vec3 e{};
              e[k] = f;
              geometric(H, rotate(b.q, bi == j.a ? j.ra : -j.rb), e);
            } else if (k < 5) {
              Vec3 aa = rotate(bodies[j.a].q, j.axisA),
                   ta = rotate(bodies[j.a].q, j.tangentA);
              if (k == 4)
                ta = cross(aa, ta);
              Vec3 ab = rotate(bodies[j.b].q, j.axisB);
              geometric(H, bi == j.a ? ta : ab, (bi == j.a ? ab : ta) * f);
            }
          }
        }
        double delta = wrap(angle(j) - j.angle0),
               stiff = j.damping / h + j.armature / (h * h),
               force = j.damping / h * delta +
                       j.armature / (h * h) * (delta - h * j.velocity0);
        if (stiff > 0)
          stamp(H, g, J[5], force, stiff);
        if (j.frictionloss > 0) {
          double trial = j.lambdaF + j.penaltyF * delta;
          stamp(H, g, J[5], std::clamp(trial, -j.frictionloss, j.frictionloss),
                std::abs(trial) < j.frictionloss ? j.penaltyF : 0);
        }
      }
      for (auto &c : contacts)
        if (c.body == bi) {
          Vec3 C;
          std::array<V6, 3> J{};
          contactValues(c, C, J);
          Vec3 f = contactForce(C, c);
          if (f.z < 0) {
            for (int k = 0; k < 3; ++k)
              stamp(H, g, J[k], f[k], c.penalty[k]);
            geometric(H, rotate(b.q, c.local), f);
          }
        }
      auto step = solve(H, g);
      Vec3 tx{step[0], step[1], step[2]}, rx{step[3], step[4], step[5]};
      double scale = std::min({1.0, 0.1 / std::max(norm(tx), 1e-30),
                               0.25 / std::max(norm(rx), 1e-30)});
      b.x += scale * tx;
      b.q = normalized(exp(scale * rx) * b.q);
      maxUpdate = std::max(maxUpdate, std::max(norm(tx), norm(rx)) * scale);
    }
    for (auto &j : joints) {
      std::array<double, 7> C{};
      std::array<V6, 7> A{}, B{};
      evaluateJoint(j, C, A, B);
      for (int k = 0; k < (j.limited ? 7 : 5); ++k) {
        double c =
            C[k] - options.alpha * (k < 5 ? j.c0[k] : std::min(j.c0[k], 0.0));
        double f = j.lambda[k] + j.penalty[k] * c;
        j.lambda[k] = k >= 5 ? std::min(f, 0.0) : f;
        if (k < 5 || f < 0)
          j.penalty[k] = std::min(j.penalty[k] + (k < 3 ? options.betaLinear
                                                        : options.betaAngular) *
                                                     std::abs(c),
                                  maximumPenalty(k));
      }
    }
    for (auto &j : joints)
      if (j.frictionloss > 0) {
        double delta = wrap(angle(j) - j.angle0);
        double trial = j.lambdaF + j.penaltyF * delta;
        j.lambdaF = std::clamp(trial, -j.frictionloss, j.frictionloss);
        if (std::abs(trial) < j.frictionloss)
          j.penaltyF =
              std::min(j.penaltyF + options.betaAngular * std::abs(delta), 1e7);
      }
    for (auto &c : contacts) {
      Vec3 C;
      std::array<V6, 3> J{};
      contactValues(c, C, J);
      Vec3 f = contactForce(C, c);
      double trial = std::hypot(c.lambda[0] + c.penalty[0] * C.x,
                                c.lambda[1] + c.penalty[1] * C.y);
      c.stick = trial <= -c.friction * f.z && std::hypot(C.x, C.y) < 1e-5;
      for (int k = 0; k < 3; ++k) {
        c.lambda[k] = f[k];
        if (f.z < 0 && (k == 2 || trial <= -c.friction * f.z))
          c.penalty[k] =
              std::min(c.penalty[k] + options.betaLinear * std::abs(C[k]), 1e9);
      }
    }
    diagnostics.update = maxUpdate;
    diagnostics.iterations = it + 1;
    // A small block update alone is insufficient to declare convergence.
    double error = 0;
    for (auto &j : joints) {
      std::array<double, 7> C{};
      std::array<V6, 7> A{}, B{};
      evaluateJoint(j, C, A, B);
      for (int k = 0; k < (j.limited ? 7 : 5); ++k) {
        double c =
            C[k] - options.alpha * (k < 5 ? j.c0[k] : std::min(j.c0[k], 0.0));
        error = std::max(error, k < 5 ? std::abs(c) : std::max(0.0, -c));
      }
    }
    for (auto &c : contacts) {
      Vec3 C;
      std::array<V6, 3> J{};
      contactValues(c, C, J);
      error = std::max(error, std::max(0.0, -C.z));
    }
    if (maxUpdate < options.tolerance && error < options.tolerance) {
      diagnostics.converged = true;
      break;
    }
  }
  for (auto &b : bodies)
    if (b.mass > 0) {
      b.v = (b.x - b.x0) / h;
      b.w = log(b.q * conjugate(b.q0)) / h;
      if (!std::isfinite(norm(b.x) + norm(b.v) + norm(b.w)))
        throw std::runtime_error("Non-finite state");
    }
  for (auto &j : joints) {
    std::array<double, 7> C{};
    std::array<V6, 7> A{}, B{};
    evaluateJoint(j, C, A, B);
    for (int k = 0; k < 3; ++k)
      diagnostics.jointError = std::max(diagnostics.jointError, std::abs(C[k]));
    for (int k = 3; k < 5; ++k)
      diagnostics.axisError = std::max(diagnostics.axisError, std::abs(C[k]));
  }
  for (auto &c : contacts)
    diagnostics.penetration = std::max(
        diagnostics.penetration,
        std::max(0.0,
                 -(bodies[c.body].x + rotate(bodies[c.body].q, c.local)).z));
  diagnostics.contacts = (int)contacts.size();
}
} // namespace duck
