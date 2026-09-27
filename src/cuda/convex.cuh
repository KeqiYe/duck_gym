/* Portal discovery/refinement adapted from libccd v2.1 src/mpr.c.
 * Copyright (c) 2010,2011 Daniel Fiser <danfis@danfis.cz>
 * BSD-3-Clause; see third_party/licenses/libccd-BSD-3-Clause.txt.
 * duck_gym adaptation: fixed storage, bounded iterations, CPU/CUDA math,
 * explicit failure status, support mapping and shape inflation.
 * Intentionally no include guard: included by kernel.cuh inside each scalar
 * namespace, after Shape and Body definitions.
 */
struct SupportPoint {
  Vec3 v{}, a{}, b{};
};
struct ConvexResult {
  // 0 separated, 1 intersecting inflated shapes, -1 indeterminate failure.
  int status = 0;
  Vec3 normal{}, position{};
  Real depth = 0;
};
HD inline Vec3 shapeSupport(const Shape &s, const Body &b, const Vec3 *points,
                            Vec3 direction, Real inflation) {
  Vec3 localDirection = rotate(conjugate(b.q), direction), local{};
  if (s.type == 0 || s.type == 2) {
    local = s.center + unit(localDirection) * s.size.x;
    if (s.type == 2) {
      Vec3 axis = rotate(s.q, {0, 0, 1});
      local += axis * (dot(axis, localDirection) >= 0 ? s.size.y : -s.size.y);
    }
  } else {
    int best = -1;
    Real value = -INFINITY;
    for (int i = lane(); i < s.count; i += stride()) {
      Real candidate = dot(points[s.offset + i], localDirection);
      if (candidate > value) {
        best = i;
        value = candidate;
      }
    }
    best = bestIndex(-value, best);
    local = points[s.offset + best];
  }
  return b.x + rotate(b.q, local) + direction * inflation;
}
struct MinkowskiQuery {
  const Shape &sa;
  const Shape &sb;
  const Body &a;
  const Body &b;
  const Vec3 *points;
  Real inflation;
  HD SupportPoint support(Vec3 direction) const {
    SupportPoint p;
    p.a = shapeSupport(sa, a, points, direction, inflation / 2);
    p.b = shapeSupport(sb, b, points, -direction, inflation / 2);
    p.v = p.a - p.b;
    return p;
  }
};
HD inline Vec3 portalDirection(const Array<SupportPoint, 4> &p) {
  return unit(cross(p[2].v - p[1].v, p[3].v - p[1].v));
}
HD inline void expandPortal(Array<SupportPoint, 4> &p, const SupportPoint &v) {
  Vec3 c = cross(v.v, p[0].v);
  if (dot(p[1].v, c) > 0) {
    if (dot(p[2].v, c) > 0)
      p[1] = v;
    else
      p[3] = v;
  } else if (dot(p[3].v, c) > 0) {
    p[2] = v;
  } else {
    p[1] = v;
  }
}
HD inline Real portalAdvance(const Array<SupportPoint, 4> &p,
                             const SupportPoint &v, Vec3 d) {
  return dot(v.v, d) -
         maximum(dot(p[1].v, d), maximum(dot(p[2].v, d), dot(p[3].v, d)));
}
// Closest point on a triangle to the origin, with explicit edge fallback.
HD inline Vec3 closestTriangleOrigin(Vec3 a, Vec3 b, Vec3 c) {
  Vec3 best = a;
  Real distance = dot(a, a);
  Array<Vec3, 3> v{{a, b, c}};
  for (int i = 0; i < 3; ++i) {
    Vec3 start = v[i], edge = v[(i + 1) % 3] - start;
    Real square = dot(edge, edge);
    Vec3 point =
        start +
        edge * (square > 0 ? clamp(-dot(start, edge) / square, 0, 1) : 0);
    if (dot(point, point) < distance) {
      distance = dot(point, point);
      best = point;
    }
  }
  Vec3 ab = b - a, ac = c - a, n = cross(ab, ac);
  Real nn = dot(n, n);
  if (nn > 0) {
    Vec3 point = n * (dot(n, a) / nn);
    Real u = dot(cross(b - point, c - point), n) / nn;
    Real v = dot(cross(c - point, a - point), n) / nn;
    if (u >= 0 && v >= 0 && u + v <= 1 && dot(point, point) < distance)
      best = point;
  }
  return best;
}
HD inline Vec3 portalPosition(const Array<SupportPoint, 4> &p, Vec3 d) {
  Array<Real, 4> w{
      {dot(cross(p[1].v, p[2].v), p[3].v), dot(cross(p[3].v, p[2].v), p[0].v),
       dot(cross(p[0].v, p[1].v), p[3].v), dot(cross(p[2].v, p[1].v), p[0].v)}};
  Real sum = w[0] + w[1] + w[2] + w[3];
  if (!(sum > 0)) {
    w[0] = 0;
    w[1] = dot(cross(p[2].v, p[3].v), d);
    w[2] = dot(cross(p[3].v, p[1].v), d);
    w[3] = dot(cross(p[1].v, p[2].v), d);
    sum = w[1] + w[2] + w[3];
  }
  if (!(sum > 0))
    return {NAN, NAN, NAN};
  Vec3 position{};
  for (int i = 0; i < 4; ++i)
    position += (p[i].a + p[i].b) * (w[i] / (2 * sum));
  return position;
}
HD inline ConvexResult convexContact(const MinkowskiQuery &query,
                                     Real tolerance) {
  ConvexResult result;
  Array<SupportPoint, 4> p;
  p[0].a = query.a.x + rotate(query.a.q, query.sa.interior);
  p[0].b = query.b.x + rotate(query.b.q, query.sb.interior);
  p[0].v = p[0].a - p[0].b;
  Real scale =
      maximum(Real(.001), query.sa.boundingRadius + query.sb.boundingRadius);
  Real epsilon = tolerance * Real(.01);
  if (norm(p[0].v) < epsilon)
    p[0].v.x += epsilon * 10;
  Vec3 direction = unit(-p[0].v);
  p[1] = query.support(direction);
  if (dot(p[1].v, direction) <= 0)
    return result;
  direction = cross(p[0].v, p[1].v);
  if (norm(direction) <= epsilon * scale) {
    result.status = 1;
    result.depth = norm(p[1].v);
    result.normal = result.depth > epsilon ? -unit(p[1].v) : unit(p[0].v);
    result.position = (p[1].a + p[1].b) / 2;
    return result;
  }
  direction = unit(direction);
  p[2] = query.support(direction);
  if (dot(p[2].v, direction) <= 0)
    return result;
  direction = unit(cross(p[1].v - p[0].v, p[2].v - p[0].v));
  if (dot(direction, p[0].v) > 0) {
    auto tmp = p[1];
    p[1] = p[2];
    p[2] = tmp;
    direction = -direction;
  }
  bool discovered = false;
  for (int iteration = 0; iteration < 128; ++iteration) {
    if (!std::isfinite(direction.x))
      break;
    p[3] = query.support(direction);
    if (dot(p[3].v, direction) <= 0)
      return result;
    if (dot(cross(p[1].v, p[3].v), p[0].v) < -epsilon * scale * scale) {
      p[2] = p[3];
    } else if (dot(cross(p[3].v, p[2].v), p[0].v) < -epsilon * scale * scale) {
      p[1] = p[3];
    } else {
      discovered = true;
      break;
    }
    direction = unit(cross(p[1].v - p[0].v, p[2].v - p[0].v));
  }
  if (!discovered) {
    result.status = -1;
    return result;
  }
  bool intersecting = false;
  for (int iteration = 0; iteration < 128; ++iteration) {
    direction = portalDirection(p);
    if (!std::isfinite(direction.x))
      break;
    if (dot(direction, p[1].v) >= -epsilon) {
      intersecting = true;
      break;
    }
    auto point = query.support(direction);
    if (dot(point.v, direction) < 0 ||
        portalAdvance(p, point, direction) <= tolerance)
      return result;
    expandPortal(p, point);
  }
  if (!intersecting) {
    result.status = -1;
    return result;
  }
  for (int iteration = 0; iteration < 256; ++iteration) {
    direction = portalDirection(p);
    if (!std::isfinite(direction.x))
      break;
    auto point = query.support(direction);
    if (portalAdvance(p, point, direction) <= tolerance) {
      Vec3 closest = closestTriangleOrigin(p[1].v, p[2].v, p[3].v);
      result.depth = norm(closest);
      result.normal =
          result.depth > epsilon ? -closest / result.depth : -direction;
      result.position = portalPosition(p, direction);
      result.status = std::isfinite(result.position.x) ? 1 : -1;
      return result;
    }
    expandPortal(p, point);
  }
  result.status = -1;
  return result;
}
