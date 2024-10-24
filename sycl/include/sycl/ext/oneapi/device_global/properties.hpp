//==----- properties.hpp - SYCL properties associated with device_global ---==//
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
#include <type_traits> // for true_type

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

template <typename T, typename PropertyListT> class device_global;

struct device_image_scope_property
    : new_properties::detail::property_base<device_image_scope_property> {
  static constexpr std::string_view property_name{
      "sycl::ext::oneapi::experimental::device_image_scope_property"};

  static constexpr const char *ir_attribute_name = "sycl-device-image-scope";
  static constexpr std::nullptr_t ir_attribute_value = nullptr;
};
inline constexpr device_image_scope_property device_image_scope;

enum class host_access_enum : std::uint16_t { read, write, read_write, none };
template <host_access_enum Access>
struct host_access_property
    : new_properties::detail::property_base<host_access_property<Access>,
                                            struct host_access_key> {
  static constexpr std::string_view property_name{
      "sycl::ext::oneapi::experimental::host_access_property"};

  static constexpr const char *ir_attribute_name = "sycl-host-access";
  static constexpr host_access_enum ir_attribute_value = Access;
};
template <host_access_enum Access>
inline constexpr host_access_property<Access> host_access;

inline constexpr auto host_access_read = host_access<host_access_enum::read>;
inline constexpr auto host_access_write = host_access<host_access_enum::write>;
inline constexpr auto host_access_read_write =
    host_access<host_access_enum::read_write>;
inline constexpr auto host_access_none = host_access<host_access_enum::none>;

enum class init_mode_enum : std::uint16_t { reprogram, reset };
template <init_mode_enum Trigger>
struct init_mode_property
    : new_properties::detail::property_base<init_mode_property<Trigger>,
                                            struct init_mode_key> {
  static constexpr std::string_view property_name{
      "sycl::ext::oneapi::experimental::init_mode_property"};

  static constexpr const char *ir_attribute_name = "sycl-init-mode";
  static constexpr init_mode_enum ir_attribute_value = Trigger;
};
template <init_mode_enum Trigger>
inline constexpr init_mode_property<Trigger> init_mode;

inline constexpr auto init_mode_reprogram =
    init_mode<init_mode_enum::reprogram>;
inline constexpr auto init_mode_reset = init_mode<init_mode_enum::reset>;

template <bool Enable>
struct implement_in_csr_property
    : new_properties::detail::property_base<implement_in_csr_property<Enable>,
                                            struct implement_in_csr_key> {
  static constexpr std::string_view property_name{
      "sycl::ext::oneapi::experimental::implement_in_csr_property"};

  static constexpr const char *ir_attribute_name = "sycl-implement-in-csr";
  static constexpr bool ir_attribute_value = Enable;
};
template <bool Enable>
inline constexpr implement_in_csr_property<Enable> implement_in_csr;

inline constexpr auto implement_in_csr_on = implement_in_csr<true>;
inline constexpr auto implement_in_csr_off = implement_in_csr<false>;

template <typename T, typename PropertyListT>
struct is_property_key_of<device_image_scope_key,
                          device_global<T, PropertyListT>> : std::true_type {};
template <typename T, typename PropertyListT>
struct is_property_key_of<host_access_key, device_global<T, PropertyListT>>
    : std::true_type {};
template <typename T, typename PropertyListT>
struct is_property_key_of<init_mode_key, device_global<T, PropertyListT>>
    : std::true_type {};
template <typename T, typename PropertyListT>
struct is_property_key_of<implement_in_csr_key, device_global<T, PropertyListT>>
    : std::true_type {};

namespace detail {
// Filter allowing additional conditions for selecting when to include meta
// information for properties for device_global.
template <typename PropT, typename Properties>
struct DeviceGlobalMetaInfoFilter : std::true_type {};

// host_access cannot be honored for device_global variables without the
// device_image_scope property, as the runtime needs to write the common USM
// pointer during first launch.
template <host_access_enum Access, typename Properties>
struct DeviceGlobalMetaInfoFilter<host_access_key::value_t<Access>, Properties>
    : std::bool_constant<
          Properties::template has_property<device_image_scope_key>()> {};

} // namespace detail
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
