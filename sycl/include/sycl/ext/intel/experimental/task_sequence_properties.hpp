//==-- task_sequence_properties.hpp - Specific properties of task_sequence -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/properties/property.hpp>       // for PropKind
#include <sycl/ext/oneapi/properties/property_value.hpp> // for property_value

#include <type_traits> // for true_type

namespace sycl {
inline namespace _V1 {
namespace ext::intel::experimental {
// Forward declare a class that these properties can be applied to
template <auto &f, typename PropertyListT> class task_sequence;

// Make sure that we are using the right namespace
template <typename PropertyT, typename... Ts>
using property_value =
    sycl::ext::oneapi::experimental::property_value<PropertyT, Ts...>;

struct balanced_property
    : ext::oneapi::experimental::new_properties::detail::property_base<
          balanced_property> {
  static constexpr std::string_view property_name{
      "sycl::ext::intel::experimental::balanced_property"};
};
inline constexpr balanced_property balanced;

template <unsigned int Size>
struct invocation_capacity_property
    : ext::oneapi::experimental::new_properties::detail::property_base<
          invocation_capacity_property<Size>, struct invocation_capacity_key> {
  static constexpr std::string_view property_name{
      "sycl::ext::intel::experimental::invocation_capacity_property"};

  static constexpr auto value = Size;
};
template <unsigned int Size>
inline constexpr invocation_capacity_property<Size> invocation_capacity;

template <unsigned int Size>
struct response_capacity_property
    : ext::oneapi::experimental::new_properties::detail::property_base<
          response_capacity_property<Size>, struct response_capacity_key> {
  static constexpr std::string_view property_name{
      "sycl::ext::intel::experimental::response_capacity_property"};

  static constexpr auto value = Size;
};
template <unsigned int Size>
inline constexpr response_capacity_property<Size> response_capacity;

} // namespace ext::intel::experimental

namespace ext::oneapi::experimental {
template <auto &f, typename PropertyListT>
struct is_property_key_of<intel::experimental::balanced_key,
                          intel::experimental::task_sequence<f, PropertyListT>>
    : std::true_type {};
template <auto &f, typename PropertyListT>
struct is_property_key_of<intel::experimental::invocation_capacity_key,
                          intel::experimental::task_sequence<f, PropertyListT>>
    : std::true_type {};
template <auto &f, typename PropertyListT>
struct is_property_key_of<intel::experimental::response_capacity_key,
                          intel::experimental::task_sequence<f, PropertyListT>>
    : std::true_type {};
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
