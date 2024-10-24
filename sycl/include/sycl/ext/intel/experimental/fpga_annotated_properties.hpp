//==-- fpga_annotated_properties.hpp - SYCL properties associated with
// annotated_arg/ptr --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/experimental/common_annotated_properties/properties.hpp>
#include <sycl/ext/oneapi/properties/properties.hpp>
#include <sycl/ext/oneapi/properties/property.hpp>
#include <sycl/ext/oneapi/properties/property_value.hpp>

#include <cstdint>
#include <iosfwd>
#include <tuple>
#include <type_traits>

namespace sycl {
inline namespace _V1 {
namespace ext {
namespace intel {
namespace experimental {

//===----------------------------------------------------------------------===//
//        FPGA properties of annotated_arg/annotated_ptr
//===----------------------------------------------------------------------===//

struct register_map_property
    : ext::oneapi::experimental::new_properties::detail::property_base<
          register_map_property> {
  static constexpr std::string_view property_name{
      "sycl::ext::intel::experimental::register_map_property"};

  static constexpr const char *ir_attribute_name = "sycl-register-map";
  static constexpr std::nullptr_t ir_attribute_value = nullptr;
};
inline constexpr register_map_property register_map;

struct conduit_property
    : ext::oneapi::experimental::new_properties::detail::property_base<
          conduit_property> {
  static constexpr std::string_view property_name{
      "sycl::ext::intel::experimental::conduit_property"};

  static constexpr const char *ir_attribute_name = "sycl-conduit";
  static constexpr std::nullptr_t ir_attribute_value = nullptr;
};
inline constexpr conduit_property conduit;

struct stable_property
    : ext::oneapi::experimental::new_properties::detail::property_base<
          stable_property> {
  static constexpr std::string_view property_name{
      "sycl::ext::intel::experimental::stable_property"};

  static constexpr const char *ir_attribute_name = "sycl-stable";
  static constexpr std::nullptr_t ir_attribute_value = nullptr;
};
inline constexpr stable_property stable;

template <int N>
struct buffer_location_property
    : ext::oneapi::experimental::new_properties::detail::property_base<
          buffer_location_property<N>, struct buffer_location_key> {
  static constexpr std::string_view property_name{
      "sycl::ext::intel::experimental::buffer_location_property"};
  static constexpr const char *ir_attribute_name = "sycl-buffer-location";
  static constexpr int ir_attribute_value = N;
};
template <int N>
inline constexpr buffer_location_property<N> buffer_location;

template <int N>
struct awidth_property
    : ext::oneapi::experimental::new_properties::detail::property_base<
          awidth_property<N>, struct awidth_key> {
  static constexpr std::string_view property_name{
      "sycl::ext::intel::experimental::awidth_property"};
  static constexpr const char *ir_attribute_name = "sycl-awidth";
  static constexpr int ir_attribute_value = N;
};
template <int N> inline constexpr awidth_property<N> awidth;

template <int K>
struct dwidth_property
    : ext::oneapi::experimental::new_properties::detail::property_base<
          dwidth_property<K>, struct dwidth_key> {
  static constexpr std::string_view property_name{
      "sycl::ext::intel::experimental::dwidth_property"};
  static constexpr const char *ir_attribute_name = "sycl-dwidth";
  static constexpr int ir_attribute_value = N;
};
template <int N> inline constexpr dwidth_property<N> dwidth;

template <int N>
struct latency_property
    : ext::oneapi::experimental::new_properties::detail::property_base<
          latency_property<N>, struct latency_key> {
  static constexpr std::string_view property_name{
      "sycl::ext::intel::experimental::latency_property"};
  static constexpr const char *ir_attribute_name = "sycl-latency";
  static constexpr int ir_attribute_value = N;
};
template <int N> inline constexpr latency_property<N> latency;

enum class read_write_mode_enum : std::uint16_t { read, write, read_write };

template <read_write_mode_enum Mode>
struct read_write_mode_property
    : ext::oneapi::experimental::new_properties::detail::property_base<
          read_write_mode_property<Mode>, struct read_write_mode_key> {
  static constexpr std::string_view property_name{
      "sycl::ext::intel::experimental::read_write_mode_property"};

  static constexpr const char *ir_attribute_name = "sycl-read-write-mode";
  static constexpr read_write_mode_enum ir_attribute_value = Mode;
};
template <read_write_mode_enum Mode>
inline constexpr read_write_mode_property<Mode> read_write_mode;
inline constexpr auto read_write_mode_read =
    read_write_mode<read_write_mode_enum::read>;
inline constexpr auto read_write_mode_write =
    read_write_mode<read_write_mode_enum::write>;
inline constexpr auto read_write_mode_read_write =
    read_write_mode<read_write_mode_enum::read_write>;

template <int N>
struct maxburst_property
    : ext::oneapi::experimental::new_properties::detail::property_base<
          maxburst_property<N>, struct maxburst_key> {
  static constexpr std::string_view property_name{
      "sycl::ext::intel::experimental::maxburst_property"};
  static constexpr const char *ir_attribute_name = "sycl-maxburst";
  static constexpr int ir_attribute_value = N;
};
template <int N> inline constexpr maxburst_property<N> maxburst;

template <int Enable>
struct wait_request_property
    : ext::oneapi::experimental::new_properties::detail::property_base<
          wait_request_property<Enable>, struct wait_request_key> {
  static constexpr std::string_view property_name{
      "sycl::ext::oneapi::experimental::wait_request_property"};

  static constexpr const char *name = "sycl-wait-request";
  static constexpr int value = Enable;
};
template <int Enable>
inline constexpr wait_request_property<Enable> wait_request;
inline constexpr auto wait_request_requested = wait_request<1>;
inline constexpr auto wait_request_not_requested = wait_request<0>;

} // namespace experimental
} // namespace intel

namespace oneapi {
namespace experimental {
template <typename T, typename PropertyListT> class annotated_arg;
template <typename T, typename PropertyListT> class annotated_ptr;

struct alignment_key;
using register_map_key = intel::experimental::register_map_key;
using conduit_key = intel::experimental::conduit_key;
using stable_key = intel::experimental::stable_key;
using buffer_location_key = intel::experimental::buffer_location_key;
using awidth_key = intel::experimental::awidth_key;
using dwidth_key = intel::experimental::dwidth_key;
using latency_key = intel::experimental::latency_key;
using read_write_mode_key = intel::experimental::read_write_mode_key;
using maxburst_key = intel::experimental::maxburst_key;
using wait_request_key = intel::experimental::wait_request_key;
using read_write_mode_enum = intel::experimental::read_write_mode_enum;

template <typename T, typename PropertyListT>
struct is_property_key_of<register_map_key, annotated_arg<T, PropertyListT>>
    : std::true_type {};

template <typename T, typename PropertyListT>
struct is_property_key_of<conduit_key, annotated_arg<T, PropertyListT>>
    : std::true_type {};

template <typename T, typename PropertyListT>
struct is_property_key_of<stable_key, annotated_arg<T, PropertyListT>>
    : std::true_type {};

template <typename T, typename PropertyListT>
struct is_property_key_of<buffer_location_key, annotated_arg<T, PropertyListT>>
    : std::true_type {};

template <typename T, typename PropertyListT>
struct is_property_key_of<awidth_key, annotated_arg<T, PropertyListT>>
    : std::true_type {};

template <typename T, typename PropertyListT>
struct is_property_key_of<dwidth_key, annotated_arg<T, PropertyListT>>
    : std::true_type {};

template <typename T, typename PropertyListT>
struct is_property_key_of<latency_key, annotated_arg<T, PropertyListT>>
    : std::true_type {};

template <typename T, typename PropertyListT>
struct is_property_key_of<read_write_mode_key, annotated_arg<T, PropertyListT>>
    : std::true_type {};

template <typename T, typename PropertyListT>
struct is_property_key_of<maxburst_key, annotated_arg<T, PropertyListT>>
    : std::true_type {};

template <typename T, typename PropertyListT>
struct is_property_key_of<wait_request_key, annotated_arg<T, PropertyListT>>
    : std::true_type {};

template <typename T, typename PropertyListT>
struct is_property_key_of<register_map_key, annotated_ptr<T, PropertyListT>>
    : std::true_type {};

template <typename T, typename PropertyListT>
struct is_property_key_of<conduit_key, annotated_ptr<T, PropertyListT>>
    : std::true_type {};

template <typename T, typename PropertyListT>
struct is_property_key_of<stable_key, annotated_ptr<T, PropertyListT>>
    : std::true_type {};

template <typename T, typename PropertyListT>
struct is_property_key_of<buffer_location_key, annotated_ptr<T, PropertyListT>>
    : std::true_type {};

template <typename T, typename PropertyListT>
struct is_property_key_of<awidth_key, annotated_ptr<T, PropertyListT>>
    : std::true_type {};

template <typename T, typename PropertyListT>
struct is_property_key_of<dwidth_key, annotated_ptr<T, PropertyListT>>
    : std::true_type {};

template <typename T, typename PropertyListT>
struct is_property_key_of<latency_key, annotated_ptr<T, PropertyListT>>
    : std::true_type {};

template <typename T, typename PropertyListT>
struct is_property_key_of<read_write_mode_key, annotated_ptr<T, PropertyListT>>
    : std::true_type {};

template <typename T, typename PropertyListT>
struct is_property_key_of<maxburst_key, annotated_ptr<T, PropertyListT>>
    : std::true_type {};

template <typename T, typename PropertyListT>
struct is_property_key_of<wait_request_key, annotated_ptr<T, PropertyListT>>
    : std::true_type {};

// 'buffer_location' and mmhost properties are pointers-only
template <typename T, int N>
struct is_valid_property<T, buffer_location_key::value_t<N>>
    : std::bool_constant<std::is_pointer_v<T>> {};

template <typename T, int W>
struct is_valid_property<T, awidth_key::value_t<W>>
    : std::bool_constant<std::is_pointer_v<T>> {};

template <typename T, int W>
struct is_valid_property<T, dwidth_key::value_t<W>>
    : std::bool_constant<std::is_pointer_v<T>> {};

template <typename T, int N>
struct is_valid_property<T, latency_key::value_t<N>>
    : std::bool_constant<std::is_pointer_v<T>> {};

template <typename T, read_write_mode_enum Mode>
struct is_valid_property<T, read_write_mode_key::value_t<Mode>>
    : std::bool_constant<std::is_pointer_v<T>> {};

template <typename T, int N>
struct is_valid_property<T, maxburst_key::value_t<N>>
    : std::bool_constant<std::is_pointer_v<T>> {};

template <typename T, int Enable>
struct is_valid_property<T, wait_request_key::value_t<Enable>>
    : std::bool_constant<std::is_pointer_v<T>> {};

// 'register_map',  'conduit',  'stable' are common properties for pointers
// and non pointers;
template <typename T>
struct is_valid_property<T, register_map_key::value_t> : std::true_type {};
template <typename T>
struct is_valid_property<T, conduit_key::value_t> : std::true_type {};
template <typename T>
struct is_valid_property<T, stable_key::value_t> : std::true_type {};

// buffer_location is applied on PtrAnnotation
template <>
struct propagateToPtrAnnotation<buffer_location_key> : std::true_type {};

//===----------------------------------------------------------------------===//
//   Utility for FPGA properties
//===----------------------------------------------------------------------===//
//
namespace detail {
template <typename... Args> struct checkValidFPGAPropertySet {
  using list = std::tuple<Args...>;
  static constexpr bool has_BufferLocation =
      ContainsProperty<buffer_location_key, list>::value;

  static constexpr bool has_InterfaceConfig =
      ContainsProperty<awidth_key, list>::value ||
      ContainsProperty<dwidth_key, list>::value ||
      ContainsProperty<latency_key, list>::value ||
      ContainsProperty<read_write_mode_key, list>::value ||
      ContainsProperty<maxburst_key, list>::value ||
      ContainsProperty<wait_request_key, list>::value;

  static constexpr bool value = !(!has_BufferLocation && has_InterfaceConfig);
};

template <typename... Args> struct checkHasConduitAndRegisterMap {
  using list = std::tuple<Args...>;
  static constexpr bool has_Conduit =
      ContainsProperty<conduit_key, list>::value;
  static constexpr bool has_RegisterMap =
      ContainsProperty<register_map_key, list>::value;
  static constexpr bool value = !(has_Conduit && has_RegisterMap);
};
} // namespace detail

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
