#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
namespace duck {
struct Vec3 {
  double x = 0, y = 0, z = 0;
  double &operator[](int i) { return i == 0 ? x : (i == 1 ? y : z); }
  double operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
};
inline Vec3 operator+(Vec3 a, Vec3 b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}
inline Vec3 operator-(Vec3 a, Vec3 b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}
inline Vec3 operator-(Vec3 a) { return {-a.x, -a.y, -a.z}; }
inline Vec3 operator*(Vec3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
inline Vec3 operator*(double s, Vec3 a) { return a * s; }
inline Vec3 operator/(Vec3 a, double s) { return a * (1 / s); }
inline Vec3 &operator+=(Vec3 &a, Vec3 b) {
  a = a + b;
  return a;
}
inline double dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(Vec3 a, Vec3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline double norm(Vec3 a) { return std::sqrt(dot(a, a)); }
inline Vec3 unit(Vec3 a) {
  double n = norm(a);
  if (n < 1e-15)
    throw std::runtime_error("Zero direction");
  return a / n;
}
struct Quat {
  double w = 1, x = 0, y = 0, z = 0;
};
inline Quat conjugate(Quat q) { return {q.w, -q.x, -q.y, -q.z}; }
inline Quat operator*(Quat a, Quat b) {
  return {a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
          a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
          a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
          a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w};
}
inline Quat normalized(Quat q) {
  double n = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
  if (n < 1e-15)
    throw std::runtime_error("Zero quaternion");
  return {q.w / n, q.x / n, q.y / n, q.z / n};
}
inline Vec3 rotate(Quat q, Vec3 v) {
  Vec3 u{q.x, q.y, q.z};
  return v + 2 * cross(u, cross(u, v) + q.w * v);
}
inline Quat exp(Vec3 v) {
  double a = norm(v), s = a < 1e-8 ? 0.5 - a * a / 48 : std::sin(a / 2) / a;
  return normalized({std::cos(a / 2), s * v.x, s * v.y, s * v.z});
}
inline Vec3 log(Quat q) {
  q = normalized(q);
  if (q.w < 0)
    q = {-q.w, -q.x, -q.y, -q.z};
  Vec3 v{q.x, q.y, q.z};
  double s = norm(v);
  return v * (s < 1e-10 ? 2 : 2 * std::atan2(s, q.w) / s);
}
inline double wrap(double a) { return std::remainder(a, 2 * std::acos(-1.0)); }
using V6 = std::array<double, 6>;
using M6 = std::array<V6, 6>;
inline V6 row(Vec3 t, Vec3 r) { return {t.x, t.y, t.z, r.x, r.y, r.z}; }
inline void stamp(M6 &H, V6 &g, const V6 &J, double force, double stiffness) {
  for (int i = 0; i < 6; ++i) {
    g[i] += J[i] * force;
    for (int j = 0; j < 6; ++j)
      H[i][j] += stiffness * J[i] * J[j];
  }
}
inline V6 solve(M6 H, V6 g) {
  // Cholesky on a positive inertia + Gauss-Newton matrix; fail explicitly on
  // bad pivots.
  M6 L{};
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j <= i; ++j) {
      double v = H[i][j];
      for (int k = 0; k < j; ++k)
        v -= L[i][k] * L[j][k];
      if (i == j) {
        if (!(v > 0) || !std::isfinite(v))
          throw std::runtime_error("Non-positive local Hessian");
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
} // namespace duck
