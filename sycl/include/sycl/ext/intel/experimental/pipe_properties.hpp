//==----- pipe_properties.hpp - SYCL properties associated with data flow pipe
//---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/properties/property.hpp>       // for PropKind
#include <sycl/ext/oneapi/properties/property_value.hpp> // for property_value

#include <cstdint>     // for uint16_t
#include <type_traits> // for true_type

namespace sycl {
inline namespace _V1 {
namespace ext::intel::experimental {

template <int Latency>
struct ready_latency_property
    : ext::oneapi::experimental::new_properties::detail::property_base<
          ready_latency_property<Latency>, struct ready_latency_key> {
  static constexpr std::string_view property_name{
      "sycl::ext::intel::experimental::ready_latency_property"};

  static constexpr auto value = Latency;
};
template <int Latency>
inline constexpr ready_latency_property<Latency> ready_latency;

template <int Bits>
struct bits_per_symbol_property
    : ext::oneapi::experimental::new_properties::detail::property_base<
          bits_per_symbol_property<Bits>, struct bits_per_symbol_key> {
  static constexpr std::string_view property_name{
      "sycl::ext::intel::experimental::bits_per_symbol_property"};

  static constexpr auto value = Bits;
};
template <int Bits>
inline constexpr bits_per_symbol_property<Bits> bits_per_symbol;

template <bool Valid>
struct uses_valid_property
    : ext::oneapi::experimental::new_properties::detail::property_base<
          uses_valid_property<Valid>, struct uses_valid_key> {
  static constexpr std::string_view property_name{
      "sycl::ext::intel::experimental::uses_valid_property"};

  static constexpr auto value = Valid;
};
template <bool Valid> inline constexpr uses_valid_property<Valid> uses_valid;
inline constexpr auto uses_valid_on = uses_valid<true>;
inline constexpr auto uses_valid_off = uses_valid<false>;

template <bool HighOrder>
struct first_symbol_in_high_order_bits_property
    : ext::oneapi::experimental::new_properties::detail::property_base<
          first_symbol_in_high_order_bits_property<HighOrder>,
          struct first_symbol_in_high_order_bits_key> {
  static constexpr std::string_view property_name{
      "sycl::ext::intel::experimental::first_symbol_in_high_order_bits_"
      "property"};

  static constexpr auto value = HighOrder;
};
template <bool HighOrder>
inline constexpr first_symbol_in_high_order_bits_property<HighOrder>
    first_symbol_in_high_order_bits;
inline constexpr auto first_symbol_in_high_order_bits_on =
    first_symbol_in_high_order_bits<true>;
inline constexpr auto first_symbol_in_high_order_bits_off =
    first_symbol_in_high_order_bits<false>;

enum class protocol_name : std::uint16_t {
  avalon_streaming = 0,
  avalon_streaming_uses_ready = 1,
  avalon_mm = 2,
  avalon_mm_uses_ready = 3
};

template <protocol_name Protocol>
struct protocol_property
    : ext::oneapi::experimental::new_properties::detail::property_base<
          protocol_property<Protocol>, struct protocol_key> {
  static constexpr std::string_view property_name{
      "sycl::ext::oneapi::experimental::"};

  static constexpr auto value = Protocol;
};
template <protocol_name Protocol>
inline constexpr protocol_property<Protocol> protocol;
// clang-format off
inline constexpr auto protocol_avalon_streaming            = protocol<protocol_name::avalon_streaming>;
inline constexpr auto protocol_avalon_streaming_uses_ready = protocol<protocol_name::avalon_streaming_uses_ready>;
inline constexpr auto protocol_avalon_mm                   = protocol<protocol_name::avalon_mm>;
inline constexpr auto protocol_avalon_mm_uses_ready        = protocol<protocol_name::avalon_mm_uses_ready>;
// clang-format on

} // namespace ext::intel::experimental
} // namespace _V1
} // namespace sycl
