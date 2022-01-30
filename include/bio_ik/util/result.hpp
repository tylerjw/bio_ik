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

#include <iostream>
#include <map>
#include <string>
#include <string_view>

#include "_external/expected.hpp"
#include "bio_ik/util/no_discard.hpp"

using tl::expected;
using tl::make_unexpected;
using tl::unexpected;

/**
 * @brief      Enum for ErrorCodes inspired by absl::StatusCode
 */
enum class ErrorCode : int {
  OK,
  CANCELLED,
  UNKNOWN,
  INVALID_ARGUMENT,
  TIMEOUT,
  NOT_FOUND,
  ALREADY_EXISTS,
  PERMISSION_DENINED,
  RESOURCE_EXHAUSTED,
  FAILED_PRECONDITION,
  ABORTED,
  OUT_OF_RANGE,
  UNIMPLEMENTED,
  INTERNAL,
  UNAVAILABLE,
  DATA_LOSS,
  UNAUTHENTICATED,
  EXCEPTION
};

/**
 * @brief      Error type used by Result<T>
 */
struct [[nodiscard]] Error {
  ErrorCode code = ErrorCode::OK;
  std::string what = "";

  inline operator bool() const noexcept { return code == ErrorCode::OK; }
  inline bool operator==(const Error& other) {
    return std::tie(code, what) == std::tie(other.code, other.what);
  }
};

constexpr auto Ok = [] { return make_unexpected(Error{ErrorCode::OK, ""}); };
constexpr auto Canceled = [](const std::string& what) {
  return make_unexpected(Error{ErrorCode::CANCELLED, what});
};
constexpr auto Unknown = [](const std::string& what) {
  return make_unexpected(Error{ErrorCode::UNKNOWN, what});
};
constexpr auto InvalidArgument = [](const std::string& what) {
  return make_unexpected(Error{ErrorCode::INVALID_ARGUMENT, what});
};
constexpr auto Timeout = [](const std::string& what) {
  return make_unexpected(Error{ErrorCode::TIMEOUT, what});
};
constexpr auto NotFound = [](const std::string& what) {
  return make_unexpected(Error{ErrorCode::NOT_FOUND, what});
};
constexpr auto AlreadyExists = [](const std::string& what) {
  return make_unexpected(Error{ErrorCode::ALREADY_EXISTS, what});
};
constexpr auto PermissionDenied = [](const std::string& what) {
  return make_unexpected(Error{ErrorCode::PERMISSION_DENINED, what});
};
constexpr auto ResourceExhausted = [](const std::string& what) {
  return make_unexpected(Error{ErrorCode::RESOURCE_EXHAUSTED, what});
};
constexpr auto FailedPrecondition = [](const std::string& what) {
  return make_unexpected(Error{ErrorCode::FAILED_PRECONDITION, what});
};
constexpr auto Aborted = [](const std::string& what) {
  return make_unexpected(Error{ErrorCode::ABORTED, what});
};
constexpr auto OutOfRange = [](const std::string& what) {
  return make_unexpected(Error{ErrorCode::OUT_OF_RANGE, what});
};
constexpr auto Unimplemented = [](const std::string& what) {
  return make_unexpected(Error{ErrorCode::UNIMPLEMENTED, what});
};
constexpr auto Internal = [](const std::string& what) {
  return make_unexpected(Error{ErrorCode::INTERNAL, what});
};
constexpr auto Unavailable = [](const std::string& what) {
  return make_unexpected(Error{ErrorCode::UNAVAILABLE, what});
};
constexpr auto DataLoss = [](const std::string& what) {
  return make_unexpected(Error{ErrorCode::DATA_LOSS, what});
};
constexpr auto Unauthenticated = [](const std::string& what) {
  return make_unexpected(Error{ErrorCode::UNAUTHENTICATED, what});
};
constexpr auto Exception = [](const std::string& what) {
  return make_unexpected(Error{ErrorCode::EXCEPTION, what});
};

/**
 * @brief      convert ErrorCode to string_view for easy formatting
 *
 * @param[in]  code  The error code
 */
[[nodiscard]] constexpr std::string_view toStringView(const ErrorCode& code) {
  switch (code) {
    case ErrorCode::OK:
      return "Ok";
    case ErrorCode::CANCELLED:
      return "Canceled";
    case ErrorCode::UNKNOWN:
      return "Unknown";
    case ErrorCode::INVALID_ARGUMENT:
      return "InvalidArgument";
    case ErrorCode::TIMEOUT:
      return "Timeout";
    case ErrorCode::NOT_FOUND:
      return "NotFound";
    case ErrorCode::ALREADY_EXISTS:
      return "AlreadyExists";
    case ErrorCode::PERMISSION_DENINED:
      return "PermissionDenied";
    case ErrorCode::RESOURCE_EXHAUSTED:
      return "ResourceExhausted";
    case ErrorCode::FAILED_PRECONDITION:
      return "FailedPrecondition";
    case ErrorCode::ABORTED:
      return "Aborted";
    case ErrorCode::OUT_OF_RANGE:
      return "OutOfRange";
    case ErrorCode::UNIMPLEMENTED:
      return "Unimplemented";
    case ErrorCode::INTERNAL:
      return "Internal";
    case ErrorCode::UNAVAILABLE:
      return "Unavailable";
    case ErrorCode::DATA_LOSS:
      return "DataLoss";
    case ErrorCode::UNAUTHENTICATED:
      return "Unauthenticated";
    case ErrorCode::EXCEPTION:
      return "Exception";
  }
}

/**
 * Result<T> type
 */
template <typename T>
using Result = expected<T, Error>;

/**
 * @brief      fmt format implementation for Error type
 */
template <>
struct fmt::formatter<Error> {
  template <typename ParseContext>
  constexpr auto parse(ParseContext& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(const Error& error, FormatContext& ctx) {
    return format_to(ctx.out(), "[Error: code={}, what={}]",
                     toStringView(error.code), error.what);
  }
};

/**
 * @brief      fmt format implementation for Result<T> type
 */
template <typename T>
struct fmt::formatter<Result<T>> {
  template <typename ParseContext>
  constexpr auto parse(ParseContext& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(const Result<T>& result, FormatContext& ctx) {
    if (result.has_value()) {
      return format_to(ctx.out(), "[Result<T>: value={}]", result.value());
    } else {
      return format_to(ctx.out(), "[Result<T>: error={}]", result.error());
    }
  }
};
