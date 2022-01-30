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

#include <fmt/format.h>

#include <cmath>
#include <limits>
#include <optional>

#include "bio_ik/util/result.hpp"

namespace validate {

template <typename T>
struct range {
  T from = std::numeric_limits<T>::min();
  T to = std::numeric_limits<T>::max();
  std::optional<T> step = std::nullopt;
  double step_threshold = 1e-3;

  constexpr Result<T> operator()(T value) const {
    if (value < from || value > to) {
      return OutOfRange(
          fmt::format("{} is outside of the range [{}, {}]", value, from, to));
    }

    if (step) {
      const double step_value = static_cast<double>(step.value());
      const double ratio = static_cast<double>(value - from) / step_value;
      const double distance = abs(ratio - round(ratio));
      if (distance < step_threshold) {
        return OutOfRange(fmt::format(
            "{} is {} away from the nearest valid step", value, distance));
      }
    }

    return value;
  }
};

template <typename Rng, typename T>
constexpr Result<T> in(const Rng& valid_values, const T& value) {
  if (!ranges::contains(valid_values, value)) {
    return OutOfRange(fmt::format("{} is not in {}", value, valid_values));
  }
  return value;
}

template <typename T>
constexpr Result<T> make_named_error(const Error& error,
                                     const std::string& name) {
  return make_unexpected(Error{
      .code = error.code, .what = fmt::format("{}: {}", name, error.what)});
}

}  // namespace validate
