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

#include <tuple>

/**
 * @brief      Curried template copied from
 *             https://gitlab.com/manning-fpcpp-book/code-examples/-/blob/master/chapter-11/make-curried/main.cpp
 *
 * @tparam     Function      The function to curry
 * @tparam     CapturedArgs  The args to capture
 */
template <typename Function, typename... CapturedArgs>
class Curried {
 private:
  using CapturedArgsTuple = std::tuple<std::decay_t<CapturedArgs>...>;

  template <typename... Args>
  static auto capture_by_copy(Args&&... args) {
    return std::tuple<std::decay_t<Args>...>(std::forward<Args>(args)...);
  }

 public:
  Curried(Function function, CapturedArgs... args)
      : m_function(function), m_captured(capture_by_copy(std::move(args)...)) {}

  Curried(Function function, std::tuple<CapturedArgs...> args)
      : m_function(function), m_captured(std::move(args)) {}

  template <typename... NewArgs>
  auto operator()(NewArgs&&... args) const {
    auto new_args = capture_by_copy(std::forward<NewArgs>(args)...);

    auto all_args = std::tuple_cat(m_captured, std::move(new_args));

    if constexpr (std::is_invocable_v<Function, CapturedArgs..., NewArgs...>) {
      return std::apply(m_function, all_args);

    } else {
      return Curried<Function, CapturedArgs..., NewArgs...>(m_function,
                                                            all_args);
    }
  }

 private:
  Function m_function;
  std::tuple<CapturedArgs...> m_captured;
};
