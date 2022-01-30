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

#include <range/v3/all.hpp>
#include <rclcpp/rclcpp.hpp>
#include <string>

#include "bio_ik/ik_evolution_1.hpp"
#include "bio_ik/ik_evolution_2.hpp"
#include "bio_ik/ik_gradient.hpp"
#include "bio_ik/ik_test.hpp"
#include "bio_ik/util/parameter_loader.hpp"
#include "bio_ik/util/result.hpp"
#include "bio_ik/util/validate.hpp"

namespace views = ::ranges::views;

namespace bio_ik {
namespace {

[[nodiscard]] auto valid_modes() {
  const auto evolution1_modes = getEvolution1Modes();
  const auto evolution2_modes = getEvolution2Modes();
  const auto gradient_decent_modes = getGradientDecentModes();
  const auto test_modes = getTestModes();

  return views::concat(
             views::all(evolution1_modes), views::all(evolution2_modes),
             views::all(gradient_decent_modes), views::all(test_modes)) |
         ranges::to<std::set<std::string>>();
}

}  // namespace

[[nodiscard]] Result<RosParameters> validate(const RosParameters& ros_params) {
  if (const auto result = validate::in(valid_modes(), ros_params.mode);
      !result) {
    return validate::make_named_error<RosParameters>(result.error(), "mode");
  }

  if (const auto result =
          validate::range<size_t>{.from = 2}(ros_params.population_size);
      !result) {
    return validate::make_named_error<RosParameters>(result.error(),
                                                     "population_size");
  }

  if (const auto result =
          validate::range<size_t>{.from = 2}(ros_params.elite_count);
      !result) {
    return validate::make_named_error<RosParameters>(result.error(),
                                                     "elite_count");
  }

  return ros_params;
}

[[nodiscard]] Result<RosParameters> get_ros_parameters(
    const rclcpp::Node::SharedPtr& node) {
  const auto default_values = RosParameters{};
  const auto loader = ParameterLoader{node};

  const auto enable_profiler =
      loader("enable_profiler", default_values.enable_profiler);
  if (!enable_profiler) return make_unexpected(enable_profiler.error());

  const auto mode = loader("mode", default_values.mode, "solver mode",
                           fmt::format("in the set: {}", valid_modes()));
  if (!mode) return make_unexpected(mode.error());

  const auto enable_counter =
      loader("enable_counter", default_values.enable_counter);
  if (!enable_counter) return make_unexpected(enable_counter.error());

  const auto random_seed = loader("random_seed", default_values.random_seed,
                                  "useful for deterministic testing");
  if (!random_seed) return make_unexpected(random_seed.error());

  const auto dpos = loader("dpos", default_values.dpos);
  if (!dpos) return make_unexpected(dpos.error());

  const auto drot = loader("drot", default_values.drot);
  if (!drot) return make_unexpected(drot.error());

  const auto dtwist = loader("dtwist", default_values.dtwist);
  if (!dtwist) return make_unexpected(dtwist.error());

  const auto skip_wipeout = loader("skip_wipeout", default_values.skip_wipeout,
                                   "used by evolution1 solvers");
  if (!skip_wipeout) return make_unexpected(skip_wipeout.error());

  const auto population_size = loader(
      "population_size", static_cast<int64_t>(default_values.population_size),
      "used by evolution1 modes", "2 or larger");
  if (!population_size) return make_unexpected(population_size.error());

  const auto elite_count =
      loader("elite_count", static_cast<int64_t>(default_values.elite_count),
             "used by evolution1 modes", "2 or larger");
  if (!elite_count) return make_unexpected(elite_count.error());

  const auto enable_linear_fitness =
      loader("enable_linear_fitness", default_values.enable_linear_fitness);
  if (!enable_linear_fitness)
    return make_unexpected(enable_linear_fitness.error());

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
