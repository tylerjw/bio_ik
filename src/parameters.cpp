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

#include "bio_ik/parameters.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <optional>
#include <range/v3/all.hpp>
#include <rcl_interfaces/msg/floating_point_range.hpp>
#include <rcl_interfaces/msg/integer_range.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <rcl_interfaces/msg/parameter_type.hpp>
#include <rclcpp/rclcpp.hpp>
#include <string>

#include "bio_ik/ik_evolution_1.hpp"
#include "bio_ik/ik_evolution_2.hpp"
#include "bio_ik/ik_gradient.hpp"
#include "bio_ik/ik_test.hpp"

namespace bio_ik {

using namespace ranges;
using Descriptor = rcl_interfaces::msg::ParameterDescriptor;
using rclcpp::ParameterValue;

class DescriptorBuilder {
  Descriptor msg_;

 public:
  DescriptorBuilder() = default;
  DescriptorBuilder(const Descriptor& msg) : msg_{msg} {}
  operator Descriptor() const { return move(msg_); }
  DescriptorBuilder description(std::string description) const {
    Descriptor msg = msg_;
    msg.description = description;
    return DescriptorBuilder{msg};
  }
  DescriptorBuilder additional_constraints(
      std::string additional_constraints) const {
    Descriptor msg = msg_;
    msg.additional_constraints = additional_constraints;
    return DescriptorBuilder{msg};
  }
  DescriptorBuilder read_only(bool read_only) const {
    Descriptor msg = msg_;
    msg.read_only = read_only;
    return DescriptorBuilder{msg};
  }
  DescriptorBuilder dynamic_typing(bool dynamic_typing) const {
    Descriptor msg = msg_;
    msg.dynamic_typing = dynamic_typing;
    return DescriptorBuilder{msg};
  }
  DescriptorBuilder floating_point_range(
      double from_value = std::numeric_limits<double>::min(),
      double to_value = std::numeric_limits<double>::max(),
      double step = 0) const {
    Descriptor msg = msg_;
    msg.floating_point_range.push_back([&] {
      rcl_interfaces::msg::FloatingPointRange range;
      range.from_value = from_value;
      range.to_value = to_value;
      range.step = step;
      return range;
    }());
    return DescriptorBuilder{msg};
  }
  DescriptorBuilder integer_range(
      int64_t from_value = std::numeric_limits<int64_t>::min(),
      int64_t to_value = std::numeric_limits<int64_t>::max(),
      uint64_t step = 0) const {
    Descriptor msg = msg_;
    msg.integer_range.push_back([&] {
      rcl_interfaces::msg::IntegerRange range;
      range.from_value = from_value;
      range.to_value = to_value;
      range.step = step;
      return range;
    }());
    return DescriptorBuilder{msg};
  }
};

std::string declare(const rclcpp::Node::SharedPtr& node,
                    const std::string& name, const std::string& default_value,
                    const Descriptor& descriptor = Descriptor()) {
  return node
      ->declare_parameter(name, ParameterValue(default_value), descriptor)
      .get<std::string>();
}

bool declare(const rclcpp::Node::SharedPtr& node, const std::string& name,
             bool default_value, const Descriptor& descriptor = Descriptor()) {
  return node
      ->declare_parameter(name, ParameterValue(default_value), descriptor)
      .get<bool>();
}

double declare(const rclcpp::Node::SharedPtr& node, const std::string& name,
               double default_value,
               const Descriptor& descriptor = Descriptor()) {
  return node
      ->declare_parameter(name, ParameterValue(default_value), descriptor)
      .get<double>();
}

int64_t declare(const rclcpp::Node::SharedPtr& node, const std::string& name,
                int64_t default_value,
                const Descriptor& descriptor = Descriptor()) {
  return node
      ->declare_parameter(name, ParameterValue(default_value), descriptor)
      .get<int64_t>();
}

size_t declare(const rclcpp::Node::SharedPtr& node, const std::string& name,
               size_t default_value,
               const Descriptor& descriptor = Descriptor()) {
  return static_cast<size_t>(
      node->declare_parameter(
              name, ParameterValue(static_cast<int64_t>(default_value)),
              descriptor)
          .get<int64_t>());
}

std::set<std::string> valid_modes() {
  const auto evolution_1 = getEvolution1ModeSet();
  const auto evolution_2 = getEvolution2ModeSet();
  const auto gradient_decent = getGradientDecentModeSet();
  const auto test = getTestModeSet();

  auto valid_modes = evolution_1;
  valid_modes.insert(evolution_2.begin(), evolution_2.end());
  valid_modes.insert(gradient_decent.begin(), gradient_decent.end());
  valid_modes.insert(test.begin(), test.begin());
  return valid_modes;
}

std::optional<std::string> validate(const RosParameters& ros_params) {
  if (valid_modes().count(ros_params.mode) == 0)
    return fmt::format("Mode: {}\nValid Modes: {}\n", ros_params.mode,
                       valid_modes());
  return std::nullopt;
}

std::optional<RosParameters> get_ros_parameters(
    const rclcpp::Node::SharedPtr& node) {
  using rclcpp::ParameterValue;
  RosParameters ros_params;

  auto ros_parameters = RosParameters{
      .enable_profiler =
          declare(node, "enable_profiler", ros_params.enable_profiler,
                  DescriptorBuilder().additional_constraints(
                      fmt::format("One of {}", valid_modes()))),
      .mode = declare(node, "mode", ros_params.mode),
      .enable_counter =
          declare(node, "enable_counter", ros_params.enable_counter),
      .random_seed = declare(node, "random_seed", ros_params.random_seed),
      .dpos = declare(node, "dpos", ros_params.dpos),
      .drot = declare(node, "drot", ros_params.drot),
      .dtwist = declare(node, "dtwist", ros_params.dtwist),
      .skip_wipeout = declare(node, "skip_wipeout", ros_params.skip_wipeout),
      .population_size =
          declare(node, "population_size", ros_params.population_size,
                  DescriptorBuilder().integer_range(2)),
      .elite_count = declare(node, "elite_count", ros_params.elite_count,
                             DescriptorBuilder().integer_range(2)),
      .enable_linear_fitness = declare(node, "enable_linear_fitness",
                                       ros_params.enable_linear_fitness),
  };

  if (auto error = validate(ros_params)) {
    RCLCPP_ERROR_STREAM(rclcpp::get_logger("bio_ik"), *error);
    return std::nullopt;
  } else {
    return ros_params;
  }
}

}  // namespace bio_ik
