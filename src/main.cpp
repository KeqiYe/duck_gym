#include "duck/solver.hpp"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
int main(int argc, char **argv) {
  try {
    if (argc < 2) {
      std::cerr << "usage: duck_sim MODEL [--steps N] [--dt H] [--iterations "
                   "N] [--output CSV] [--stride N] [--alpha A]\n";
      return 2;
    }
    duck::Solver sim;
    sim.load(argv[1]);
    int steps = 1000, stride = 1;
    std::string output;
    for (int i = 2; i < argc; ++i) {
      std::string arg = argv[i];
      if (++i == argc)
        throw std::runtime_error("Missing option value");
      std::string value = argv[i];
      if (arg == "--steps")
        steps = std::stoi(value);
      else if (arg == "--dt")
        sim.options.dt = std::stod(value);
      else if (arg == "--iterations")
        sim.options.iterations = std::stoi(value);
      else if (arg == "--alpha")
        sim.options.alpha = std::stod(value);
      else if (arg == "--output")
        output = value;
      else if (arg == "--stride")
        stride = std::stoi(value);
      else
        throw std::runtime_error("Unknown option: " + arg);
    }
    if (steps < 1 || stride < 1)
      throw std::runtime_error("Steps and stride must be positive");
    std::ofstream out;
    if (!output.empty()) {
      out.open(output);
      if (!out)
        throw std::runtime_error("Cannot open output");
      out << std::setprecision(17)
          << "step,time,body,x,y,z,qw,qx,qy,qz,vx,vy,vz,wx,wy,wz\n";
    }
    auto write = [&](int step) {
      if (out.is_open())
        for (int b = 1; b < (int)sim.bodies.size(); ++b) {
          auto &v = sim.bodies[b];
          out << step << ',' << step * sim.options.dt << ',' << b << ','
              << v.x.x << ',' << v.x.y << ',' << v.x.z << ',' << v.q.w << ','
              << v.q.x << ',' << v.q.y << ',' << v.q.z << ',' << v.v.x << ','
              << v.v.y << ',' << v.v.z << ',' << v.w.x << ',' << v.w.y << ','
              << v.w.z << '\n';
        }
    };
    write(0);
    double elapsed = 0, maxJoint = 0, maxAxis = 0, maxPenetration = 0,
           iterationSum = 0;
    int failures = 0, maxContacts = 0;
    for (int step = 1; step <= steps; ++step) {
      auto start = std::chrono::steady_clock::now();
      sim.step();
      elapsed += std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - start)
                     .count();
      auto &d = sim.diagnostics;
      maxJoint = std::max(maxJoint, d.jointError);
      maxAxis = std::max(maxAxis, d.axisError);
      maxPenetration = std::max(maxPenetration, d.penetration);
      maxContacts = std::max(maxContacts, d.contacts);
      iterationSum += d.iterations;
      if (!d.converged)
        ++failures;
      if (step % stride == 0 || step == steps)
        write(step);
    }
    std::cout << std::setprecision(12)
              << "{\"backend\":\"cpu_fp64\",\"steps\":" << steps
              << ",\"dt\":" << sim.options.dt
              << ",\"iterations_budget\":" << sim.options.iterations
              << ",\"mean_iterations\":" << iterationSum / steps
              << ",\"seconds\":" << elapsed
              << ",\"steps_per_second\":" << steps / elapsed
              << ",\"joint_error_m\":" << maxJoint
              << ",\"axis_error\":" << maxAxis
              << ",\"penetration_m\":" << maxPenetration
              << ",\"max_contacts\":" << maxContacts
              << ",\"unconverged_steps\":" << failures << "}\n";
  } catch (const std::exception &e) {
    std::cerr << "duck_sim: " << e.what() << '\n';
    return 1;
  }
}
