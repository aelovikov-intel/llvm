//==---------------- kernel_properties.hpp - SYCL Kernel Properties -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// APIs for setting kernel properties interpreted by GPU software stack.
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp>
#include <sycl/ext/oneapi/properties/property.hpp>
#include <sycl/ext/oneapi/properties/property_value.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {
enum class register_alloc_mode_enum : uint32_t {
  automatic = 0,
  large = 2,
};

template <register_alloc_mode_enum Mode>
struct register_alloc_mode_property
    : ext::oneapi::experimental::new_properties::detail::property_base<
          register_alloc_mode_property<Mode>, struct register_alloc_mode_key> {
  static constexpr std::string_view property_name{
      "sycl::ext::oneapi::experimental::"};

  static constexpr const char *ir_attribute_name = "sycl-register-alloc-mode";
  static constexpr sycl::detail::register_alloc_mode_enum ir_attribute_value =
      Mode;
};
template <register_alloc_mode_enum Mode>
inline constexpr register_alloc_mode_property<Mode> register_alloc_mode
    __SYCL_DEPRECATED("register_alloc_mode is deprecated, "
                      "use sycl::ext::intel::experimental::grf_size or "
                      "sycl::ext::intel::experimental::grf_size_automatic");
} // namespace detail
} // namespace _V1
} // namespace sycl
