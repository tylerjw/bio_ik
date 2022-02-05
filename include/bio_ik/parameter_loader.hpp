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

#include <fp/all.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <rcl_interfaces/msg/parameter_type.hpp>
#include <rclcpp/rclcpp.hpp>
#include <string>

/**
 * @brief      Callable that declares and gets parameters using rclcpp.
 */
class ParameterLoader {
  rclcpp::Node::SharedPtr node_;

  [[nodiscard]] rcl_interfaces::msg::ParameterDescriptor make_descriptor(
      std::string const& description, std::string const& constraints) const {
    rcl_interfaces::msg::ParameterDescriptor msg;
    msg.description = description;
    msg.additional_constraints = constraints;
    return msg;
  }

  template <typename T>
  [[nodiscard]] fp::Result<T> declare_parameter(
      std::string const& name, T const& default_value,
      std::string const& description = "",
      std::string const& constraints = "") const {
    return fp::try_to_result([&] {
      return node_
          ->declare_parameter(name, rclcpp::ParameterValue{default_value},
                              make_descriptor(description, constraints))
          .get<T>();
    });
  }

  template <typename T>
  [[nodiscard]] fp::Result<T> get_parameter(std::string const& name) const {
    return fp::try_to_result(
        [&] { return node_->get_parameter(name).get_value<T>(); });
  }

 public:
  ParameterLoader(const rclcpp::Node::SharedPtr& node) : node_{node} {}

  template <typename T>
  [[nodiscard]] fp::Result<T> operator()(
      std::string const& name, T const& default_value,
      std::string const& description = "",
      std::string const& constraints = "") const {
    if (!node_->has_parameter(name)) {
      if (auto const result = declare_parameter<T>(name, default_value,
                                                   description, constraints);
          !result) {
        return tl::make_unexpected(result.error());
      }
    }

    return get_parameter<T>(name);
  }
};
