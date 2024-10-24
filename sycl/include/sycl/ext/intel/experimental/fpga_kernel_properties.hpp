//===--------------------- fpga_kernel_properties.hpp ---------------------===//
// SYCL properties associated with FPGA kernel properties
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/properties/property.hpp>
#include <sycl/ext/oneapi/properties/property_value.hpp>

#include <cstdint>
#include <type_traits>

namespace sycl {
inline namespace _V1 {
namespace ext::intel::experimental {

template <typename T, typename PropertyListT> class fpga_kernel_attribute;
template <auto &f, typename PropertyListT> class task_sequence;

enum class streaming_interface_options_enum : uint16_t {
  accept_downstream_stall,
  remove_downstream_stall
};

enum class register_map_interface_options_enum : uint16_t {
  do_not_wait_for_done_write,
  wait_for_done_write,
};

enum class fpga_cluster_options_enum : std::uint16_t {
  stall_free,
  stall_enable
};

template <streaming_interface_options_enum option>
struct streaming_interface_property
    : ext::oneapi::experimental::new_properties::detail::property_base<
          streaming_interface_property<option>,
          struct streaming_interface_key> {
  static constexpr std::string_view property_name{
      "sycl::ext::intel::experimental::streaming_interface_property"};

  static constexpr const char *ir_attribute_name = "sycl-streaming-interface";
  static constexpr auto ir_attribute_value = option;
};
template <streaming_interface_options_enum option =
              streaming_interface_options_enum::accept_downstream_stall>
inline constexpr streaming_interface_property<option> streaming_interface;

inline constexpr auto streaming_interface_accept_downstream_stall =
    streaming_interface<
        streaming_interface_options_enum::accept_downstream_stall>;

inline constexpr auto streaming_interface_remove_downstream_stall =
    streaming_interface<
        streaming_interface_options_enum::remove_downstream_stall>;

template <register_map_interface_options_enum option>
struct register_map_interface_property
    : ext::oneapi::experimental::new_properties::detail::property_base<
          register_map_interface_property<option>,
          struct register_map_interface_key> {
  static constexpr std::string_view property_name{
      "sycl::ext::intel::experimental::register_map_interface_property"};

  static constexpr const char *ir_attribute_name =
      "sycl-register-map-interface";
  static constexpr auto ir_attribute_value = option;
};
template <register_map_interface_options_enum option =
              register_map_interface_options_enum::do_not_wait_for_done_write>
inline constexpr register_map_interface_property<option> register_map_interface;

inline constexpr auto register_map_interface_wait_for_done_write =
    register_map_interface<
        register_map_interface_options_enum::wait_for_done_write>;

inline constexpr auto register_map_interface_do_not_wait_for_done_write =
    register_map_interface<
        register_map_interface_options_enum::do_not_wait_for_done_write>;

template <int pipeline_directive_or_initiation_interval>
struct pipelined_property
    : ext::oneapi::experimental::new_properties::detail::property_base<
          pipelined_property<pipeline_directive_or_initiation_interval>,
          struct pipelined_key> {
  static constexpr std::string_view property_name{
      "sycl::ext::intel::experimental::pipelined_property"};

  static constexpr const char *ir_attribute_name = "sycl-pipelined";
  static constexpr int ir_attribute_value =
      pipeline_directive_or_initiation_interval;
};
template <int pipeline_directive_or_initiation_interval = -1>
inline constexpr pipelined_property<pipeline_directive_or_initiation_interval>
    pipelined;

template <fpga_cluster_options_enum option>
struct fpga_cluster_property
    : ext::oneapi::experimental::new_properties::detail::property_base<
          fpga_cluster_property, struct fpga_cluster_key> {
  static constexpr std::string_view property_name{
      "sycl::ext::intel::experimental::fpga_cluster_property"};

  static constexpr const char *ir_attribute_name = "sycl-fpga-cluster";
  static constexpr auto ir_attribute_value = option;
};

template <fpga_cluster_options_enum option =
              fpga_cluster_options_enum::stall_free>
inline constexpr fpga_cluster_property<option> fpga_cluster;

inline constexpr auto stall_free_clusters =
    fpga_cluster<fpga_cluster_options_enum::stall_free>;

inline constexpr auto stall_enable_clusters =
    fpga_cluster<fpga_cluster_options_enum::stall_enable>;

} // namespace ext::intel::experimental

namespace ext::oneapi::experimental {
template <typename T, typename PropertyListT>
struct is_property_key_of<
    intel::experimental::streaming_interface_key,
    intel::experimental::fpga_kernel_attribute<T, PropertyListT>>
    : std::true_type {};
template <typename T, typename PropertyListT>
struct is_property_key_of<
    intel::experimental::register_map_interface_key,
    intel::experimental::fpga_kernel_attribute<T, PropertyListT>>
    : std::true_type {};
template <typename T, typename PropertyListT>
struct is_property_key_of<
    intel::experimental::pipelined_key,
    intel::experimental::fpga_kernel_attribute<T, PropertyListT>>
    : std::true_type {};

template <typename T, typename PropertyListT>
struct is_property_key_of<
    intel::experimental::fpga_cluster_key,
    intel::experimental::fpga_kernel_attribute<T, PropertyListT>>
    : std::true_type {};

template <auto &f, typename PropertyListT>
struct is_property_key_of<intel::experimental::pipelined_key,
                          intel::experimental::task_sequence<f, PropertyListT>>
    : std::true_type {};

template <auto &f, typename PropertyListT>
struct is_property_key_of<intel::experimental::fpga_cluster_key,
                          intel::experimental::task_sequence<f, PropertyListT>>
    : std::true_type {};

namespace detail {
template <intel::experimental::streaming_interface_options_enum option>
struct HasCompileTimeEffect<
    intel::experimental::streaming_interface_key::value_t<option>>
    : std::true_type {};
template <intel::experimental::register_map_interface_options_enum option>
struct HasCompileTimeEffect<
    intel::experimental::register_map_interface_key::value_t<option>>
    : std::true_type {};

} // namespace detail
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
