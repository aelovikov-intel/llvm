//==- grf_size_properties.hpp - GRF size kernel properties for Intel GPUs -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/kernel_properties.hpp>
#include <sycl/ext/oneapi/properties/properties.hpp>
#include <sycl/ext/oneapi/properties/property.hpp>
#include <sycl/ext/oneapi/properties/property_value.hpp>

#define SYCL_EXT_INTEL_GRF_SIZE 1

namespace sycl {
inline namespace _V1 {
namespace ext::intel::experimental {

template <unsigned int Size>
struct grf_size_property : oneapi::experimental::detail::property_base<
                               grf_size_property<Size>,
                               oneapi::experimental::detail::PropKind::GRFSize,
                               struct grf_size_common_key> {
  static_assert(Size == 128 || Size == 256, "Unsupported GRF size");
  static constexpr const char *ir_attribute_name = "sycl-grf-size";
  static constexpr unsigned int ir_attribute_value = Size;
};
using grf_size_key = grf_size_common_key;

struct grf_size_automatic_property
    : oneapi::experimental::detail::property_base<
          grf_size_automatic_property,
          // Could reuse GRFSize PropKind as well.
          oneapi::experimental::detail::PropKind::GRFSizeAutomatic,
          struct grf_size_common_key> {
  static constexpr const char *ir_attribute_name = "sycl-grf-size";
  static constexpr unsigned int ir_attribute_value = 0;
};
using grf_size_automatic_key = grf_size_common_key;

template <unsigned int Size>
inline constexpr grf_size_property<Size> grf_size;

inline constexpr grf_size_automatic_property grf_size_automatic;

} // namespace ext::intel::experimental
namespace ext::oneapi::experimental::detail {
template <typename Properties>
struct ConflictingProperties<sycl::ext::intel::experimental::grf_size_common_key,
                             Properties>
    : std::bool_constant<
          Properties::template has_property<
              sycl::detail::register_alloc_mode_key>()> {};

// Technically, the above is enough.
template <typename Properties>
struct ConflictingProperties<sycl::detail::register_alloc_mode_key, Properties>
    : std::bool_constant<Properties::template has_property<
          sycl::ext::intel::experimental::grf_size_common_key>()> {};

} // namespace ext::oneapi::experimental::detail
} // namespace _V1
} // namespace sycl
