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

#include <cfloat>
#include <limits>
#include <optional>
#include <random>
#include <rclcpp/rclcpp.hpp>
#include <string>

namespace bio_ik {

/**
 * @brief      Parameters settable via ros.
 */
struct RosParameters {
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
};

/**
 * @brief      Validates a ros_params struct
 *
 * @param[in]  ros_params  The ros parameters struct
 *
 * @return     error string if invalid, nullopt if valid
 */
std::optional<std::string> validate(const RosParameters& ros_params);

/**
 * @brief      Gets the ros parameters
 *
 * @param[in]  node  The ros node
 *
 * @return     The ros parameters on success, nullopt otherwise
 */
std::optional<RosParameters> get_ros_parameters(
    const rclcpp::Node::SharedPtr& node);

}  // namespace bio_ik
