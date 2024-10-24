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
struct grf_size_property
    : ext::oneapi::experimental::new_properties::detail::property_base<
          grf_size_property<Size>, struct grf_size_key> {
  static constexpr std::string_view property_name{
      "sycl::ext::oneapi::experimental::grf_size_property"};

  static_assert(Size == 128 || Size == 256, "Unsupported GRF size");
  static constexpr const char *ir_attribute_name = "sycl-grf-size";
  static constexpr unsigned int ir_attribute_value = Size;
};
template <unsigned int Size>
inline constexpr grf_size_property<Size> grf_size;

// TODO: Why not just
//
//   inline constexpr auto grf_size_automatic = grf_size<0>;
//
struct grf_size_automatic_property
    : ext::oneapi::experimental::new_properties::detail::property_base<
          grf_size_automatic_property> {
  static constexpr std::string_view property_name{
      "sycl::ext::oneapi::experimental::grf_size_automatic_property"};

  static constexpr const char *ir_attribute_name = "sycl-grf-size";
  static constexpr unsigned int ir_attribute_value = 0;
};
inline constexpr grf_size_automatic_property grf_size_automatic;

} // namespace ext::intel::experimental
namespace ext::oneapi::experimental::detail {

template <typename Properties>
struct ConflictingProperties<sycl::ext::intel::experimental::grf_size_key,
                             Properties>
    : std::bool_constant<
          ContainsProperty<
              sycl::ext::intel::experimental::grf_size_automatic_key,
              Properties>::value ||
          ContainsProperty<sycl::detail::register_alloc_mode_key,
                           Properties>::value> {};

template <typename Properties>
struct ConflictingProperties<
    sycl::ext::intel::experimental::grf_size_automatic_key, Properties>
    : std::bool_constant<
          ContainsProperty<sycl::ext::intel::experimental::grf_size_key,
                           Properties>::value ||
          ContainsProperty<sycl::detail::register_alloc_mode_key,
                           Properties>::value> {};

template <typename Properties>
struct ConflictingProperties<sycl::detail::register_alloc_mode_key, Properties>
    : std::bool_constant<
          ContainsProperty<sycl::ext::intel::experimental::grf_size_key,
                           Properties>::value ||
          ContainsProperty<
              sycl::ext::intel::experimental::grf_size_automatic_key,
              Properties>::value> {};

} // namespace ext::oneapi::experimental::detail
} // namespace _V1
} // namespace sycl
