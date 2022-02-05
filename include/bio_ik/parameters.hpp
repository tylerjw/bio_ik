// Copyright (c) 2022, Tyler Weaver
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//
//    * Neither the name of the Universität Hamburg nor the names of its
//      contributors may be used to endorse or promote products derived from
//      this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <fmt/core.h>

#include <cfloat>
#include <fp/all.hpp>
#include <limits>
#include <random>
#include <rclcpp/rclcpp.hpp>
#include <string>

namespace bio_ik {

/**
 * @brief      Parameters settable via ros.
 */
struct [[nodiscard]] RosParameters {
  // Plugin parameters
  bool enable_profiler = false;

  // IKParallel parameters
  std::string mode = "bio2_memetic";
  bool enable_counter = false;
  int64_t random_seed = static_cast<int64_t>(std::random_device()());

  // Problem parameters
  double dpos = DBL_MAX;
  double drot = DBL_MAX;
  double dtwist = 1e-5;

  // ik_evolution_1 parameters
  bool skip_wipeout = false;
  size_t population_size = 8;
  size_t elite_count = 4;
  bool enable_linear_fitness = false;

  inline operator std::string() const {
    return fmt::format(
        "[bio_ik::RosParameters:\n  enable_profiler={},\n  mode={},\n  "
        "enable_counter={},\n  random_seed={},\n  dpos={},\n  drot={},\n  "
        "dtwist={},\n  skip_wipeout={},\n  population_size={},\n  "
        "elite_count={},\n  enable_linear_fitness={},\n]",
        enable_profiler, mode, enable_counter, random_seed, dpos, drot, dtwist,
        skip_wipeout, population_size, elite_count, enable_linear_fitness);
  }
};

/**
 * @brief      Validates a ros_params struct
 *
 * @param[in]  ros_params  The ros parameters struct
 *
 * @return     The ros parameters on success, error status otherwise
 */
[[nodiscard]] fp::Result<RosParameters> validate(
    RosParameters const& ros_params);

/**
 * @brief      Gets the ros parameters
 *
 * @param[in]  node  The ros node
 *
 * @return     The ros parameters on success, error status otherwise
 */
[[nodiscard]] fp::Result<RosParameters> get_ros_parameters(
    rclcpp::Node::SharedPtr const& node);

}  // namespace bio_ik
