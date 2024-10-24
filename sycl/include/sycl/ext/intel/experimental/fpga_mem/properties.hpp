//==----- properties.hpp - SYCL properties associated with fpga_mem ---==//
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
#include <iosfwd>      // for nullptr_t
#include <string_view> // for string_view
#include <type_traits> // for true_type

namespace sycl {
inline namespace _V1 {
namespace ext {
namespace intel::experimental {

// Forward declare a class that these properties can be applied to
template <typename T, typename PropertyListT> class fpga_mem;

// Make sure that we are using the right namespace
template <typename PropertyT, typename... Ts>
using property_value =
    sycl::ext::oneapi::experimental::property_value<PropertyT, Ts...>;

// Property definitions
enum class resource_enum : std::uint16_t { mlab, block_ram };

template <resource_enum Resource>
struct resource_property
    : ext::oneapi::experimental::new_properties::detail::property_base<
          resource_property<Resource>, struct resource_key> {
  static constexpr std::string_view property_name{
      "sycl::ext::intel::experimental::resource_property"};

  static constexpr const char *ir_attribute_name = "sycl-resource";
  static constexpr const char *ir_attribute_value =
      ((Resource == resource_enum::mlab) ? "MLAB" : "BLOCK_RAM");
};
template <resource_enum R> inline constexpr resource_property<R> resource;
inline constexpr auto resource_mlab = auto resource<resource_enum::mlab>;
inline constexpr auto resource_block_ram =
    auto resource<resource_enum::block_ram>;

template <size_t Elements>
struct num_banks_property
    : ext::oneapi::experimental::new_properties::detail::property_base<
          num_banks_property<Elements>, struct num_banks_key> {
  static constexpr std::string_view property_name{
      "sycl::ext::intel::experimental::num_banks_property"};

  static constexpr const char *ir_attribute_name = "sycl-num-banks";
  static constexpr size_t ir_attribute_value = Elements;
};
template <size_t Elements>
inline constexpr num_banks_property<Elements> num_banks;

template <size_t Elements>
struct stride_size_property
    : ext::oneapi::experimental::new_properties::detail::property_base<
          stride_size_property<Elements>, struct stride_size_key> {
  static constexpr std::string_view property_name{
      "sycl::ext::intel::experimental::stride_size_property"};

  static constexpr const char *ir_attribute_name = "sycl-stride-size";
  static constexpr size_t ir_attribute_value = Elements;
};
template <size_t Elements>
inline constexpr stride_size_property<Elements> stride_size;

template <size_t Elements>
struct word_size_property
    : ext::oneapi::experimental::new_properties::detail::property_base<
          word_size_property<Elements>, struct word_size_key> {
  static constexpr std::string_view property_name{
      "sycl::ext::intel::experimental::word_size_property"};

  static constexpr const char *ir_attribute_name = "sycl-word-size";
  static constexpr size_t ir_attribute_value = Elements;
};
template <size_t Elements>
inline constexpr word_size_property<Elements> word_size;

template <bool Enable>
struct bi_directional_ports_property
    : ext::oneapi::experimental::new_properties::detail::property_base<
          bi_directional_ports_property<Enable>,
          struct bi_directional_ports_key> {
  static constexpr std::string_view property_name{
      "sycl::ext::intel::experimental::bi_directional_ports_property"};

  // historical uglyness: single property maps to different SPIRV decorations
  static constexpr const char *ir_attribute_name =
      (Enable ? "sycl-bi-directional-ports-true"
              : "sycl-bi-directional-ports-false");
  static constexpr std::nullptr_t ir_attribute_value = nullptr;
};
template <bool Enable>
inline constexpr bi_directional_ports_property<Enable> bi_directional_ports;
inline constexpr auto bi_directional_ports_false = bi_directional_ports<false>;
inline constexpr auto bi_directional_ports_true = bi_directional_ports<true>;

template <bool Enable>
struct clock_2x_property
    : ext::oneapi::experimental::new_properties::detail::property_base<
          clock_2x_property<Enable>, struct clock_2x_key> {
  static constexpr std::string_view property_name{
      "sycl::ext::intel::experimental::clock_2x_property"};

  // historical uglyness: single property maps to different SPIRV decorations
  static constexpr const char *ir_attribute_name =
      (Enable ? "sycl-clock-2x-true" : "sycl-clock-2x-false");
  static constexpr std::nullptr_t ir_attribute_value = nullptr;
};
template <bool Enable> inline constexpr clock_2x_property<Enable> clock_2x;
inline constexpr auto clock_2x_true = clock_2x<true>;
inline constexpr auto clock_2x_false = clock_2x<false>;

enum class ram_stitching_enum : std::uint16_t { min_ram, max_fmax };

template <ram_stitching_enum RamStitching>
struct ram_stitching_property
    : ext::oneapi::experimental::new_properties::detail::property_base<
          ram_stitching_property<RamStitching>, struct ram_stitching_key> {
  static constexpr std::string_view property_name{
      "sycl::ext::intel::experimental::ram_stitching_property"};

  static constexpr const char *ir_attribute_name = "sycl-ram-stitching";
  // enum to bool conversion to match with the SPIR-V decoration
  // ForcePow2DepthINTEL
  static constexpr size_t ir_attribute_value =
      static_cast<size_t>(RamStitching == ram_stitching_enum::max_fmax);
};
template <ram_stitching_enum RamStitching>
inline constexpr ram_stitching_property<RamStitching> ram_stitching;
inline constexpr auto ram_stitching_min_ram =
    ram_stitching<ram_stitching_enum::min_ram>;
inline constexpr auto ram_stitching_max_fmax =
    ram_stitching<ram_stitching_enum::max_fmax>;

template <size_t N>
struct max_private_copies_property
    : ext::oneapi::experimental::new_properties::detail::property_base<
          max_private_copies_property<N>, struct max_private_copies_key> {
  static constexpr std::string_view property_name{
      "sycl::ext::intel::experimental::max_private_copies_property"};

  static constexpr const char *ir_attribute_name = "sycl-max-private-copies";
  static constexpr size_t ir_attribute_value = N;
};
template <size_t N>
inline constexpr max_private_copies_property<N> max_private_copies;

template <size_t N>
struct num_replicates_property
    : ext::oneapi::experimental::new_properties::detail::property_base<
          num_replicates_property<N>, struct num_replicates_key> {
  static constexpr std::string_view property_name{
      "sycl::ext::intel::experimental::num_replicates_property"};

  static constexpr const char *ir_attribute_name = "sycl-num-replicates";
  static constexpr size_t ir_attribute_value = N;
};
template <size_t N> inline constexpr num_replicates_property<N> num_replicates;

} // namespace intel::experimental

namespace oneapi::experimental {

// Associate properties with fpga_mem
template <typename T, typename PropertyListT>
struct is_property_key_of<intel::experimental::resource_key,
                          intel::experimental::fpga_mem<T, PropertyListT>>
    : std::true_type {};
template <typename T, typename PropertyListT>
struct is_property_key_of<intel::experimental::num_banks_key,
                          intel::experimental::fpga_mem<T, PropertyListT>>
    : std::true_type {};
template <typename T, typename PropertyListT>
struct is_property_key_of<intel::experimental::stride_size_key,
                          intel::experimental::fpga_mem<T, PropertyListT>>
    : std::true_type {};
template <typename T, typename PropertyListT>
struct is_property_key_of<intel::experimental::word_size_key,
                          intel::experimental::fpga_mem<T, PropertyListT>>
    : std::true_type {};
template <typename T, typename PropertyListT>
struct is_property_key_of<intel::experimental::bi_directional_ports_key,
                          intel::experimental::fpga_mem<T, PropertyListT>>
    : std::true_type {};
template <typename T, typename PropertyListT>
struct is_property_key_of<intel::experimental::clock_2x_key,
                          intel::experimental::fpga_mem<T, PropertyListT>>
    : std::true_type {};
template <typename T, typename PropertyListT>
struct is_property_key_of<intel::experimental::ram_stitching_key,
                          intel::experimental::fpga_mem<T, PropertyListT>>
    : std::true_type {};
template <typename T, typename PropertyListT>
struct is_property_key_of<intel::experimental::max_private_copies_key,
                          intel::experimental::fpga_mem<T, PropertyListT>>
    : std::true_type {};
template <typename T, typename PropertyListT>
struct is_property_key_of<intel::experimental::num_replicates_key,
                          intel::experimental::fpga_mem<T, PropertyListT>>
    : std::true_type {};

} // namespace oneapi::experimental
} // namespace ext
} // namespace _V1
} // namespace sycl
