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

#include <fp/all.hpp>
#include <range/v3/all.hpp>
#include <rclcpp/rclcpp.hpp>
#include <string>

#include "bio_ik/ik_evolution_1.hpp"
#include "bio_ik/ik_evolution_2.hpp"
#include "bio_ik/ik_gradient.hpp"
#include "bio_ik/ik_test.hpp"
#include "bio_ik/parameter_loader.hpp"

namespace views = ::ranges::views;

namespace bio_ik {
namespace {

[[nodiscard]] auto valid_modes() {
  auto const evolution1_modes = getEvolution1Modes();
  auto const evolution2_modes = getEvolution2Modes();
  auto const gradient_decent_modes = getGradientDecentModes();
  auto const test_modes = getTestModes();

  return views::concat(
             views::all(evolution1_modes), views::all(evolution2_modes),
             views::all(gradient_decent_modes), views::all(test_modes)) |
         ranges::to<std::set<std::string>>();
}

}  // namespace

[[nodiscard]] fp::Result<RosParameters> validate(
    RosParameters const& ros_params) {
  if (auto const result = fp::validate_in(valid_modes(), ros_params.mode);
      !result) {
    return tl::make_unexpected(fp::make_named(result.error(), "mode"));
  }

  if (auto const result =
          fp::validate_range<size_t>{.from = 2}(ros_params.population_size);
      !result) {
    return tl::make_unexpected(
        fp::make_named(result.error(), "population_size"));
  }

  if (auto const result =
          fp::validate_range<size_t>{.from = 2}(ros_params.elite_count);
      !result) {
    return tl::make_unexpected(fp::make_named(result.error(), "elite_count"));
  }

  return ros_params;
}

[[nodiscard]] fp::Result<RosParameters> get_ros_parameters(
    rclcpp::Node::SharedPtr const& node) {
  auto const default_values = RosParameters{};
  auto const loader = ParameterLoader{node};

  auto const enable_profiler =
      loader("enable_profiler", default_values.enable_profiler);
  auto const mode = loader("mode", default_values.mode, "solver mode",
                           fmt::format("in the set: {}", valid_modes()));
  auto const enable_counter =
      loader("enable_counter", default_values.enable_counter);
  auto const random_seed = loader("random_seed", default_values.random_seed,
                                  "useful for deterministic testing");
  auto const dpos = loader("dpos", default_values.dpos);
  auto const drot = loader("drot", default_values.drot);
  auto const dtwist = loader("dtwist", default_values.dtwist);
  auto const skip_wipeout = loader("skip_wipeout", default_values.skip_wipeout,
                                   "used by evolution1 solvers");
  auto const population_size = loader(
      "population_size", static_cast<int64_t>(default_values.population_size),
      "used by evolution1 modes", "2 or larger");
  auto const elite_count =
      loader("elite_count", static_cast<int64_t>(default_values.elite_count),
             "used by evolution1 modes", "2 or larger");
  auto const enable_linear_fitness =
      loader("enable_linear_fitness", default_values.enable_linear_fitness);

  if (auto const error =
          fp::maybe_error(enable_profiler, mode, enable_counter, random_seed,
                          dpos, drot, dtwist, skip_wipeout, population_size,
                          elite_count, enable_linear_fitness);
      error) {
    return tl::make_unexpected(*error);
  }

  return validate(RosParameters{
      .enable_profiler = enable_profiler.value(),
      .mode = mode.value(),
      .enable_counter = enable_counter.value(),
      .random_seed = random_seed.value(),
      .dpos = dpos.value(),
      .drot = drot.value(),
      .dtwist = dtwist.value(),
      .skip_wipeout = skip_wipeout.value(),
      .population_size = static_cast<size_t>(population_size.value()),
      .elite_count = static_cast<size_t>(elite_count.value()),
      .enable_linear_fitness = enable_linear_fitness.value(),
  });
}

}  // namespace bio_ik
