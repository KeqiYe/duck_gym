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
// Included twice with float/double namespaces. CPU reference and CUDA use this
// identical implementation; the original FP64 Solver remains an independent
// oracle.
namespace DUCK_NAMESPACE {
using Real = DUCK_SCALAR;
#define HD __host__ __device__
template <class T, int N> struct Array {
  T values[N]{};
  HD T &operator[](int i) { return values[i]; }
  HD const T &operator[](int i) const { return values[i]; }
  HD T *begin() { return values; }
  HD T *end() { return values + N; }
  HD const T *begin() const { return values; }
  HD const T *end() const { return values + N; }
};
HD inline Real minimum(Real a, Real b) { return a < b ? a : b; }
HD inline Real maximum(Real a, Real b) { return a > b ? a : b; }
HD inline Real clamp(Real v, Real a, Real b) {
  return minimum(maximum(v, a), b);
}
HD inline int lane() {
#ifdef __CUDA_ARCH__
  return threadIdx.x;
#else
  return 0;
#endif
}
HD inline int stride() {
#ifdef __CUDA_ARCH__
  return blockDim.x;
#else
  return 1;
#endif
}
HD inline void synchronize() {
#ifdef __CUDA_ARCH__
  __syncthreads();
#endif
}
HD inline Real blockMaximum(Real v) {
#ifdef __CUDA_ARCH__
  for (int offset = 16; offset; offset /= 2)
    v = maximum(v, __shfl_down_sync(0xffffffff, v, offset));
  return __shfl_sync(0xffffffff, v, 0);
#else
  return v;
#endif
}
HD inline int bestIndex(Real value, int index) {
#ifdef __CUDA_ARCH__
  for (int offset = 16; offset; offset /= 2) {
    Real other = __shfl_down_sync(0xffffffff, value, offset);
    int k = __shfl_down_sync(0xffffffff, index, offset);
    if (other < value ||
        (other == value && k >= 0 && (index < 0 || k < index))) {
      value = other;
      index = k;
    }
  }
  return __shfl_sync(0xffffffff, index, 0);
#else
  return index;
#endif
}
struct Vec3 {
  Real x = 0, y = 0, z = 0;
  HD Real &operator[](int i) { return i == 0 ? x : (i == 1 ? y : z); }
  HD Real operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
};
HD inline Vec3 operator+(Vec3 a, Vec3 b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}
HD inline Vec3 operator-(Vec3 a, Vec3 b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}
HD inline Vec3 operator-(Vec3 a) { return {-a.x, -a.y, -a.z}; }
HD inline Vec3 operator*(Vec3 a, Real s) { return {a.x * s, a.y * s, a.z * s}; }
HD inline Vec3 operator*(Real s, Vec3 a) { return a * s; }
HD inline Vec3 operator/(Vec3 a, Real s) { return a * (1 / s); }
HD inline Vec3 &operator+=(Vec3 &a, Vec3 b) {
  a = a + b;
  return a;
}
HD inline Real dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
HD inline Vec3 cross(Vec3 a, Vec3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
HD inline Real norm(Vec3 a) { return std::sqrt(dot(a, a)); }
HD inline Vec3 unit(Vec3 a) {
  Real n = norm(a);
  if (n < 1e-15)
    return {NAN, NAN, NAN};
  return a / n;
}
struct Quat {
  Real w = 1, x = 0, y = 0, z = 0;
};
HD inline Quat conjugate(Quat q) { return {q.w, -q.x, -q.y, -q.z}; }
HD inline Quat operator*(Quat a, Quat b) {
  return {a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
          a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
          a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
          a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w};
}
HD inline Quat normalized(Quat q) {
  Real n = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
  if (n < 1e-15)
    return {NAN, NAN, NAN, NAN};
  return {q.w / n, q.x / n, q.y / n, q.z / n};
}
HD inline Vec3 rotate(Quat q, Vec3 v) {
  Vec3 u{q.x, q.y, q.z};
  return v + 2 * cross(u, cross(u, v) + q.w * v);
}
HD inline Quat exp(Vec3 v) {
  Real a = norm(v), s = a < 1e-8 ? 0.5 - a * a / 48 : std::sin(a / 2) / a;
  return normalized({std::cos(a / 2), s * v.x, s * v.y, s * v.z});
}
HD inline Vec3 log(Quat q) {
  q = normalized(q);
  if (q.w < 0)
    q = {-q.w, -q.x, -q.y, -q.z};
  Vec3 v{q.x, q.y, q.z};
  Real s = norm(v);
  return v * (s < 1e-10 ? 2 : 2 * std::atan2(s, q.w) / s);
}
HD inline Real wrap(Real a) { return std::remainder(a, 2 * std::acos(-1.0)); }
using V6 = Array<Real, 6>;
using M6 = Array<V6, 6>;
HD inline V6 row(Vec3 t, Vec3 r) { return {t.x, t.y, t.z, r.x, r.y, r.z}; }
HD inline void stamp(M6 &H, V6 &g, const V6 &J, Real force, Real stiffness) {
  for (int i = 0; i < 6; ++i) {
    g[i] += J[i] * force;
    for (int j = 0; j < 6; ++j)
      H[i][j] += stiffness * J[i] * J[j];
  }
}
HD inline V6 solve(M6 H, V6 g) {
  // Cholesky on a positive inertia + Gauss-Newton matrix; fail explicitly on
  // bad pivots.
  M6 L{};
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j <= i; ++j) {
      Real v = H[i][j];
      for (int k = 0; k < j; ++k)
        v -= L[i][k] * L[j][k];
      if (i == j) {
        if (!(v > 0) || !std::isfinite(v))
          return {NAN, NAN, NAN, NAN, NAN, NAN};
        L[i][j] = std::sqrt(v);
      } else
        L[i][j] = v / L[j][j];
    }
  V6 y{}, x{};
  for (int i = 0; i < 6; ++i) {
    y[i] = -g[i];
    for (int k = 0; k < i; ++k)
      y[i] -= L[i][k] * y[k];
    y[i] /= L[i][i];
  }
  for (int i = 5; i >= 0; --i) {
    x[i] = y[i];
    for (int k = i + 1; k < 6; ++k)
      x[i] -= L[k][i] * x[k];
    x[i] /= L[i][i];
  }
  return x;
}
struct Body {
  Real mass = 0;
  Vec3 inertia{}, x{}, v{}, w{}, x0{}, xp{}, force{};
  Quat q{}, q0{}, qp{};
};
struct Joint {
  int a = 0, b = 0;
  Vec3 ra{}, rb{}, axisA{}, axisB{}, tangentA{}, tangentB{};
  Real lo = 0, hi = 0, damping = 0, armature = 0, frictionloss = 0, lambdaF = 0,
       penaltyF = 1, torque = 0, kp = 0, target = 0, maxTorque = 0;
  bool limited = false;
  Real angle0 = 0, velocity0 = 0;
  Array<Real, 7> lambda{}, penalty{}, c0{};
};
struct Shape {
  int body = 0, type = 0, offset = 0, count = 0;
  Vec3 center{}, size{};
  Quat q{};
  Real friction = .8, boundingRadius = 0;
};
struct Contact {
  int body = 0, shape = 0, feature = 0;
  Vec3 local{}, anchor{};
  Real radius = 0, friction = 0, c0 = 0;
  Array<Real, 3> lambda{}, penalty{};
  bool stick = false;
};
struct Options {
  Real dt = 0.001, alpha = 0.9, gamma = 0.99, betaLinear = 1e7,
       betaAngular = 1e5, tolerance = 1e-7;
  int iterations = 100;
  Vec3 gravity{0, 0, -9.81};
  bool ground = false;
};
struct Diagnostics {
  Real jointError = 0, axisError = 0, penetration = 0, update = 0;
  int contacts = 0, iterations = 0;
  bool converged = false;
};
HD Vec3 inertia(const Body &b, Vec3 w, bool inverse = false) {
  Vec3 local = rotate(conjugate(b.q0), w);
  for (int k = 0; k < 3; ++k)
    local[k] *= inverse ? 1 / b.inertia[k] : b.inertia[k];
  return rotate(b.q0, local);
}
// Positive diagonal majorization of rotational geometric stiffness (AVBD
// Sec. 3.5).
HD void geometric(M6 &H, Vec3 r, Vec3 f) {
  Real rf = dot(r, f);
  for (int col = 0; col < 3; ++col) {
    Real sum = 0;
    for (int row = 0; row < 3; ++row) {
      Real a =
          0.5 * (r[row] * f[col] + f[row] * r[col]) - (row == col ? rf : 0);
      sum += a * a;
    }
    H[col + 3][col + 3] += std::sqrt(sum);
  }
}
HD Real minimumPenalty(int k) { return k < 3 ? 1000 : 1; }
HD Real maximumPenalty(int k) { return k < 3 ? 1e9 : 1e7; }
HD Vec3 contactForce(Vec3 C, const Contact &c) {
  Vec3 f;
  for (int k = 0; k < 3; ++k)
    f[k] = c.lambda[k] + c.penalty[k] * C[k];
  f.z = minimum(f.z, 0.0);
  Real t = std::hypot(f.x, f.y), bound = -c.friction * f.z;
  if (t > bound && t > 0) {
    f.x *= bound / t;
    f.y *= bound / t;
  }
  return f;
}
template <class T, int N> struct List : Array<T, N> {
  int length = 0;
  HD int size() const { return length; }
  HD T *end() { return this->values + length; }
  HD const T *end() const { return this->values + length; }
  HD void clear() { length = 0; }
  HD void push_back(const T &v) { this->values[length++] = v; }
};
struct Model {
  List<Body, 32> bodies;
  List<Joint, 32> joints;
  List<Shape, 32> shapes;
  Options options;
  Array<int, 32> colors;
};
struct State {
  using Scalar = Real;
  using ModelType = Model;
  using Point = Vec3;
  List<Body, 32> bodies;
  List<Joint, 32> joints;
  List<Contact, 128> contacts, previous;
  const Model *model;
  const Vec3 *points;
  Options options;
  Diagnostics diagnostics;
  int failed = 0;
  HD void initialize(const Model *m, const Vec3 *p) {
    model = m;
    points = p;
    options = m->options;
    bodies = m->bodies;
    joints = m->joints;
    contacts.clear();
    diagnostics = {};
    failed = 0;
  }
  HD void resetPose(const Real *q, const Real *roots) {
    for (auto &b : bodies) {
      b.v = {};
      b.w = {};
      b.force = {};
    }
    bodies[1].x += Vec3{roots[0], roots[1], roots[2]};
    bodies[1].q = exp({roots[3], roots[4], roots[5]}) * bodies[1].q;
    for (int i = 0; i < joints.size(); ++i) {
      auto &j = joints[i];
      const auto &origA = model->bodies[j.a];
      const auto &origB = model->bodies[j.b];
      Vec3 originalAxis = rotate(origA.q, j.axisA),
           t = rotate(origA.q, j.tangentA), u = rotate(origB.q, j.tangentB);
      Real originalAngle =
          std::atan2(dot(originalAxis, cross(t, u)), dot(t, u));
      auto &a = bodies[j.a];
      auto &b = bodies[j.b];
      Vec3 axis = rotate(a.q, j.axisA);
      b.q = normalized(exp(axis * (q[i] - originalAngle)) * a.q *
                       conjugate(origA.q) * origB.q);
      b.x = a.x + rotate(a.q, j.ra) - rotate(b.q, j.rb);
    }
  }
  HD void advance(int substeps) {
    Diagnostics summary{};
    summary.converged = true;
    for (int k = 0; k < substeps; ++k) {
      step();
      synchronize();
      if (failed)
        break;
      if (lane() == 0) {
        summary.jointError =
            maximum(summary.jointError, diagnostics.jointError);
        summary.axisError = maximum(summary.axisError, diagnostics.axisError);
        summary.penetration =
            maximum(summary.penetration, diagnostics.penetration);
        summary.contacts = maximum(summary.contacts, diagnostics.contacts);
        summary.converged = summary.converged && diagnostics.converged;
        summary.iterations += diagnostics.iterations;
      }
    }
    if (lane() == 0)
      diagnostics = summary;
    synchronize();
  }
  HD void contactForces(Real *out) const {
    for (int i = 0; i < bodies.size() * 3; ++i)
      out[i] = 0;
    for (auto &c : contacts)
      for (int k = 0; k < 3; ++k)
        out[c.body * 3 + k] -= c.lambda[k];
  }
  HD void snapshot(Real *b, Real *j, Real *d) const {
    for (int i = 0; i < bodies.size(); ++i) {
      auto &v = bodies[i];
      Real row[] = {v.x.x, v.x.y, v.x.z, v.q.w, v.q.x, v.q.y, v.q.z,
                    v.v.x, v.v.y, v.v.z, v.w.x, v.w.y, v.w.z};
      for (int k = 0; k < 13; ++k)
        b[13 * i + k] = row[k];
    }
    for (int i = 0; i < joints.size(); ++i) {
      auto &v = joints[i];
      j[2 * i] = angle(v);
      j[2 * i + 1] =
          dot(bodies[v.b].w - bodies[v.a].w, rotate(bodies[v.a].q, v.axisA));
    }
    d[0] = diagnostics.jointError;
    d[1] = diagnostics.penetration;
    d[2] = diagnostics.contacts;
    d[3] = diagnostics.converged;
    d[4] = diagnostics.axisError;
    d[5] = failed;
  }
  HD Vec3 point(const Shape &s, const Body &b, int k) const {
    if (s.type == 1 || s.type == 3)
      return points[s.offset + k];
    Vec3 down = rotate(conjugate(b.q), {0, 0, -s.size.x});
    if (s.type == 0)
      return s.center + down;
    return s.center + rotate(s.q, {0, 0, (k ? 1 : -1) * s.size.y}) + down;
  }
  HD void detectContacts() {
    if (lane() == 0) {
      previous = contacts;
      contacts.clear();
    }
    synchronize();
    if (!options.ground)
      return;
    for (int si = 0; si < model->shapes.size(); ++si) {
      const auto &s = model->shapes[si];
      const auto &b = bodies[s.body];
      if (b.mass == 0)
        continue;
      Real low =
          minimum(b.x.z, b.x.z + options.dt * b.v.z +
                             options.dt * options.dt *
                                 (options.gravity.z + b.force.z / b.mass));
      if (low - s.boundingRadius * (1 + options.dt * norm(b.w)) >= .003)
        continue;
      Array<Vec3, 4> chosen{};
      int n = 0;
      for (int pick = 0; pick < 4; ++pick) {
        int best = -1;
        Real lowest = INFINITY;
        Vec3 bestWorld{}, bestLocal{};
        for (int k = lane(); k < s.count; k += stride()) {
          Vec3 local = point(s, b, k), world = b.x + rotate(b.q, local);
          Vec3 vel = b.v + cross(b.w, world - b.x);
          Real z = minimum(world.z,
                           world.z + options.dt * vel.z +
                               options.dt * options.dt *
                                   (options.gravity.z + b.force.z / b.mass));
          if (z >= .003 || z >= lowest)
            continue;
          bool duplicate = false;
          for (int c = 0; c < n; ++c)
            if (std::hypot(chosen[c].x - world.x, chosen[c].y - world.y) < .005)
              duplicate = true;
          if (duplicate)
            continue;
          best = k;
          lowest = z;
          bestWorld = world;
          bestLocal = local;
        }
        best = bestIndex(lowest, best);
        if (best < 0)
          break;
        bestLocal = point(s, b, best);
        bestWorld = b.x + rotate(b.q, bestLocal);
        if (lane() == 0) {
          Contact c{};
          c.body = s.body;
          c.shape = si;
          c.feature = best;
          c.local = bestLocal;
          c.anchor = {bestWorld.x, bestWorld.y, 0};
          c.friction = s.friction;
          c.penalty = {1000, 1000, 1000};
          for (auto &cached : previous)
            if (cached.shape == si && cached.feature == best) {
              c.lambda = cached.lambda;
              c.penalty = cached.penalty;
              if (cached.stick) {
                c.local = cached.local;
                c.anchor = cached.anchor;
              }
              break;
            }
          c.c0 = minimum((b.x + rotate(b.q, c.local)).z, Real(0));
          for (int d = 0; d < 3; ++d) {
            c.lambda[d] *= options.alpha * options.gamma;
            c.penalty[d] =
                clamp(c.penalty[d] * options.gamma, Real(1000), Real(1e9));
          }
          contacts.push_back(c);
        }
        chosen[n++] = bestWorld;
      }
    }
  }
  HD Real angle(const Joint &j) const {
    const auto &a = bodies[j.a];
    const auto &b = bodies[j.b];
    Vec3 axis = rotate(a.q, j.axisA), t = rotate(a.q, j.tangentA),
         u = rotate(b.q, j.tangentB);
    return std::atan2(dot(axis, cross(t, u)), dot(t, u));
  }
  HD void evaluateJoint(const Joint &j, Array<Real, 7> &C, Array<V6, 7> &JA,
                        Array<V6, 7> &JB) const {
    const auto &a = bodies[j.a];
    const auto &b = bodies[j.b];
    Vec3 ra = rotate(a.q, j.ra), rb = rotate(b.q, j.rb),
         d = a.x + ra - b.x - rb;
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
    Real q = angle(j);
    C[5] = q - j.lo;
    C[6] = j.hi - q;
    JA[5] = row({}, -axis);
    JB[5] = row({}, axis);
    JA[6] = row({}, axis);
    JB[6] = row({}, -axis);
  }
  HD void contactValues(const Contact &c, Vec3 &C, Array<V6, 3> &J) const {
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
  HD Real updateBody(int bi) {
    const Real h = options.dt;
    auto &b = bodies[bi];
    if (b.mass == 0)
      return 0;
    M6 H{};
    V6 g{};
    Vec3 dx = b.x - b.xp, theta = log(b.q * conjugate(b.qp));
    for (int k = 0; k < 3; ++k) {
      H[k][k] = b.mass / (h * h);
      g[k] = H[k][k] * dx[k];
    }
    // SO(3) log Jacobian under left perturbations, with frozen world inertia.
    Real a = norm(theta),
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
    for (int ji = 0; ji < joints.size(); ++ji) {
      if (joints[ji].a != bi && joints[ji].b != bi)
        continue;
      auto &j = joints[ji];
      Array<Real, 7> C{};
      Array<V6, 7> A{}, B{};
      evaluateJoint(j, C, A, B);
      const auto &J = bi == j.a ? A : B;
      for (int k = 0; k < (j.limited ? 7 : 5); ++k) {
        Real c =
            C[k] - options.alpha * (k < 5 ? j.c0[k] : minimum(j.c0[k], 0.0));
        Real f = j.lambda[k] + j.penalty[k] * c;
        if (k >= 5)
          f = minimum(f, 0.0);
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
      // Implicit motor potential: derivative of a torque-limited PD spring.
      // Together with armature this avoids a large free-body torque predictor.
      Real drive = j.kp * wrap(angle(j) - j.target) - j.torque;
      Real driveStiffness = j.kp;
      if (j.maxTorque > 0) {
        if (std::abs(drive) >= j.maxTorque)
          driveStiffness = 0;
        drive = clamp(drive, -j.maxTorque, j.maxTorque);
      }
      stamp(H, g, J[5], drive, driveStiffness);
      Real delta = wrap(angle(j) - j.angle0),
           stiff = j.damping / h + j.armature / (h * h),
           force = j.damping / h * delta +
                   j.armature / (h * h) * (delta - h * j.velocity0);
      if (stiff > 0)
        stamp(H, g, J[5], force, stiff);
      if (j.frictionloss > 0) {
        Real trial = j.lambdaF + j.penaltyF * delta;
        stamp(H, g, J[5], clamp(trial, -j.frictionloss, j.frictionloss),
              std::abs(trial) < j.frictionloss ? j.penaltyF : 0);
      }
    }
    for (auto &c : contacts)
      if (c.body == bi) {
        Vec3 C;
        Array<V6, 3> J{};
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
    Real scale =
        minimum(Real(1), minimum(Real(.1) / maximum(norm(tx), Real(1e-30)),
                                 Real(.25) / maximum(norm(rx), Real(1e-30))));
    b.x += scale * tx;
    b.q = normalized(exp(scale * rx) * b.q);
    return maximum(norm(tx), norm(rx)) * scale;
  }
  HD void step() {
    const Real h = options.dt;
    if (!(h > 0) || options.iterations < 1 || options.alpha < 0 ||
        options.alpha >= 1 || options.gamma <= 0 || options.gamma >= 1) {
      failed = 1;
      return;
    }
    if (lane() == 0)
      diagnostics = {};
    detectContacts();
    synchronize();
    if (lane() == 0) {
      for (auto &j : joints) {
        Array<V6, 7> A{}, B{};
        evaluateJoint(j, j.c0, A, B);
        j.angle0 = angle(j);
        j.lambdaF *= options.alpha * options.gamma;
        j.penaltyF = clamp(j.penaltyF * options.gamma, 1.0, 1e7);
        Vec3 axis = rotate(bodies[j.a].q, j.axisA);
        j.velocity0 = dot(bodies[j.b].w - bodies[j.a].w, axis);
        for (int k = 0; k < 7; ++k) {
          j.lambda[k] *= options.alpha * options.gamma;
          j.penalty[k] = clamp(j.penalty[k] * options.gamma, minimumPenalty(k),
                               maximumPenalty(k));
        }
      }
      for (int i = 0; i < (int)bodies.size(); ++i) {
        auto &b = bodies[i];
        b.x0 = b.x;
        b.q0 = b.q;
        if (b.mass == 0)
          continue;
        b.xp = b.x + h * b.v + h * h * (options.gravity + b.force / b.mass);
        Vec3 wp = b.w + h * inertia(b, -cross(b.w, inertia(b, b.w)), true);
        b.qp = exp(h * wp) * b.q;
        b.x = b.xp;
        b.q = b.qp;
      }
    }
    synchronize();
    for (int it = 0; it < options.iterations; ++it) {
      Real maxUpdate = 0;
      for (int color = 0; color < 2; ++color) {
        for (int bi = lane(); bi < bodies.size(); bi += stride())
          if (model->colors[bi] == color)
            maxUpdate = maximum(maxUpdate, updateBody(bi));
        synchronize();
      }
      maxUpdate = blockMaximum(maxUpdate);
      for (int ji = lane(); ji < joints.size(); ji += stride()) {
        auto &j = joints[ji];
        Array<Real, 7> C{};
        Array<V6, 7> A{}, B{};
        evaluateJoint(j, C, A, B);
        for (int k = 0; k < (j.limited ? 7 : 5); ++k) {
          Real c =
              C[k] - options.alpha * (k < 5 ? j.c0[k] : minimum(j.c0[k], 0.0));
          Real f = j.lambda[k] + j.penalty[k] * c;
          j.lambda[k] = k >= 5 ? minimum(f, 0.0) : f;
          if (k < 5 || f < 0)
            j.penalty[k] =
                minimum(j.penalty[k] +
                            (k < 3 ? options.betaLinear : options.betaAngular) *
                                std::abs(c),
                        maximumPenalty(k));
        }
      }
      for (int ji = lane(); ji < joints.size(); ji += stride()) {
        auto &j = joints[ji];
        if (j.frictionloss > 0) {
          Real delta = wrap(angle(j) - j.angle0);
          Real trial = j.lambdaF + j.penaltyF * delta;
          j.lambdaF = clamp(trial, -j.frictionloss, j.frictionloss);
          if (std::abs(trial) < j.frictionloss)
            j.penaltyF = minimum(
                j.penaltyF + options.betaAngular * std::abs(delta), 1e7);
        }
      }
      for (int ci = lane(); ci < contacts.size(); ci += stride()) {
        auto &c = contacts[ci];
        Vec3 C;
        Array<V6, 3> J{};
        contactValues(c, C, J);
        Vec3 f = contactForce(C, c);
        Real trial = std::hypot(c.lambda[0] + c.penalty[0] * C.x,
                                c.lambda[1] + c.penalty[1] * C.y);
        c.stick = trial <= -c.friction * f.z && std::hypot(C.x, C.y) < 1e-5;
        for (int k = 0; k < 3; ++k) {
          c.lambda[k] = f[k];
          if (f.z < 0 && (k == 2 || trial <= -c.friction * f.z))
            c.penalty[k] = minimum(
                c.penalty[k] + options.betaLinear * std::abs(C[k]), 1e9);
        }
      }
      // A small block update alone is insufficient to declare convergence.
      Real error = 0;
      for (int ji = lane(); ji < joints.size(); ji += stride()) {
        auto &j = joints[ji];
        Array<Real, 7> C{};
        Array<V6, 7> A{}, B{};
        evaluateJoint(j, C, A, B);
        for (int k = 0; k < (j.limited ? 7 : 5); ++k) {
          Real c =
              C[k] - options.alpha * (k < 5 ? j.c0[k] : minimum(j.c0[k], 0.0));
          error = maximum(error, k < 5 ? std::abs(c) : maximum(0.0, -c));
        }
      }
      for (int ci = lane(); ci < contacts.size(); ci += stride()) {
        auto &c = contacts[ci];
        Vec3 C;
        Array<V6, 3> J{};
        contactValues(c, C, J);
        error = maximum(error, maximum(0.0, -C.z));
      }
      error = blockMaximum(error);
      if (lane() == 0) {
        diagnostics.update = maxUpdate;
        diagnostics.iterations = it + 1;
        diagnostics.converged =
            maxUpdate < options.tolerance && error < options.tolerance;
      }
      synchronize();
      if (diagnostics.converged)
        break;
    }
    if (lane() == 0) {
      for (auto &b : bodies)
        if (b.mass > 0) {
          b.v = (b.x - b.x0) / h;
          b.w = log(b.q * conjugate(b.q0)) / h;
          if (!std::isfinite(norm(b.x) + norm(b.v) + norm(b.w))) {
            failed = 1;
            return;
          }
        }
      for (auto &j : joints) {
        Array<Real, 7> C{};
        Array<V6, 7> A{}, B{};
        evaluateJoint(j, C, A, B);
        for (int k = 0; k < 3; ++k)
          diagnostics.jointError =
              maximum(diagnostics.jointError, std::abs(C[k]));
        for (int k = 3; k < 5; ++k)
          diagnostics.axisError =
              maximum(diagnostics.axisError, std::abs(C[k]));
      }
      for (auto &c : contacts)
        diagnostics.penetration = maximum(
            diagnostics.penetration,
            maximum(0.0,
                    -(bodies[c.body].x + rotate(bodies[c.body].q, c.local)).z));
      diagnostics.contacts = (int)contacts.size();
    }
  }
};
#undef HD
} // namespace DUCK_NAMESPACE
