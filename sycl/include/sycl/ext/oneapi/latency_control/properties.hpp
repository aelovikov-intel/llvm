//==--- properties.hpp - SYCL properties associated with latency_control ---==//
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

enum class latency_control_type {
  none, // default
  exact,
  max,
  min
};

template <int Anchor>
struct latency_anchor_id_property
    : oneapi::experimental::new_properties::detail::property_base<
          latency_anchor_id_property<Anchor>, struct latency_anchor_id_key> {
  static constexpr std::string_view property_name{
      "sycl::ext::intel::experimental::latency_anchor_id_property"};

  static constexpr auto value = latency_control;
};
template <int Anchor>
inline constexpr latency_anchor_id_property<Anchor> latency_anchor_id;

template <int Target, latency_control_type Type, int Cycle>
struct latency_constraint_property
    : oneapi::experimental::new_properties::detail::property_base<
          latency_constraint_property<Target, Type, Cycle>,
          struct latency_constraint_key> {
  static constexpr std::string_view property_name{
      "sycl::ext::intel::experimental::latency_constraint_property"};

  static constexpr int target = Target;
  static constexpr intel::experimental::latency_control_type type = Type;
  static constexpr int cycle = Cycle;
};

} // namespace ext::intel::experimental
} // namespace _V1
} // namespace sycl
