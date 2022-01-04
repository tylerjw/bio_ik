// Copyright 2021 PickNik Inc
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
//    * Neither the name of the PickNik Inc nor the names of its
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

#include <iostream>
#include <memory>
#include <rclcpp/rclcpp.hpp>

#include "bio_ik/parameters.hpp"
#include "gtest/gtest.h"

TEST(ParameterTests, DefaultValidMode)  // NOLINT
{
  // GIVEN a default constructed bio_ik::RosParameters
  const auto ros_params = bio_ik::RosParameters{};

  // WHEN we call validate
  const auto status = bio_ik::validate(ros_params);

  // THEN we expect the status to not contain an error
  EXPECT_TRUE(status) << "bio_ik::validate returned error: " << status.what;
}

TEST(ParameterTests, NotValidMode)  // NOLINT
{
  // GIVEN a bio_ik::RosParameters with an invalid mode
  const auto ros_params = bio_ik::RosParameters{.mode = "invalid"};

  // WHEN we call validate
  const auto status = bio_ik::validate(ros_params);

  // THEN we expect the status to contain an error
  EXPECT_FALSE(status) << "bio_ik::validate did not return an error";
}

TEST(ParameterTests, GetIsValid)  // NOLIINT
{
  // WHEN we call get_ros_parameters with no overridden parameters
  const auto ros_params =
      bio_ik::get_ros_parameters(std::make_shared<rclcpp::Node>("_"));

  // THEN we expect the result to be valid
  EXPECT_TRUE(ros_params) << "get_ros_parameters returned error: "
                          << ros_params.error().what;
}

TEST(ParameterTests, DefaultSameAsRosDefault)  // NOLINT
{
  // GIVEN a default constructed bio_ik::RosParameters
  const auto default_ros_parameters = bio_ik::RosParameters{};

  // WHEN we create another one using the get_ros_parameters function without
  // overriding any ros parameters
  const auto ros_params =
      bio_ik::get_ros_parameters(std::make_shared<rclcpp::Node>("_"));

  // THEN we expect the result to have a value and that to be the same as the
  // default constructed one except for the random seed
  ASSERT_TRUE(ros_params) << "get_ros_parameters returned error: "
                          << ros_params.error().what;
  EXPECT_TRUE([](const auto& lhs, const auto& rhs) {
    return std::tie(lhs.enable_profiler, lhs.mode, lhs.enable_counter, lhs.dpos,
                    lhs.drot, lhs.dtwist, lhs.skip_wipeout, lhs.population_size,
                    lhs.elite_count, lhs.enable_linear_fitness) ==
           std::tie(rhs.enable_profiler, rhs.mode, rhs.enable_counter, rhs.dpos,
                    rhs.drot, rhs.dtwist, rhs.skip_wipeout, rhs.population_size,
                    rhs.elite_count, rhs.enable_linear_fitness);
  }(ros_params.value(), default_ros_parameters))
      << "\nget_ros_parameters: " + std::string(ros_params.value()) +
             "\ndefault_ros_parameters: " + std::string(default_ros_parameters);
}

TEST(ParameterTests, StringOperator)  // NOLINT
{
  // GIVEN a default constructed bio_ik::RosParameters
  // WHEN we convert it to std::string
  // THEN we don't expect it to not throw
  EXPECT_NO_THROW(auto _ = std::string(bio_ik::RosParameters{}));
}

TEST(ParameterTests, GetInvalidParameters)  // NOLINT
{
  // GIVEN a ros node with a parameter override
  auto node = std::make_shared<rclcpp::Node>(
      "_", rclcpp::NodeOptions().append_parameter_override(
               "mode", rclcpp::ParameterValue("invalid")));

  // WHEN we call get_ros_parameters
  auto result = bio_ik::get_ros_parameters(node);

  // THEN we expect the result to have failed
  EXPECT_FALSE(result)
      << "get_ros_parameters should have returned no value, returned: " +
             std::string(result.value());
}

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);

  int result = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return result;
}
