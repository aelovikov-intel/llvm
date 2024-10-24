//==------- properties.hpp - SYCL properties associated with kernels -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <array>                                             // for array
#include <limits>
#include <stddef.h>                                          // for size_t
#include <stdint.h>                                          // for uint32_T
#include <sycl/aspects.hpp>                                  // for aspect
#include <sycl/ext/oneapi/experimental/forward_progress.hpp> // for forward_progress_guarantee enum
#include <sycl/ext/oneapi/properties/properties.hpp>
#include <type_traits>                                   // for true_type
#include <utility>                                       // for declval
namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {
namespace detail {
// Trait for checking that all size_t values are non-zero.
template <size_t... Xs> struct AllNonZero {
  static constexpr bool value = true;
};
template <size_t X, size_t... Xs> struct AllNonZero<X, Xs...> {
  static constexpr bool value = X > 0 && AllNonZero<Xs...>::value;
};
} // namespace detail

struct properties_tag {};

template <size_t... Dims>
struct work_group_size_property
    : new_properties::detail::property_base<work_group_size_property<Dims...>,
                                            struct work_group_size_key> {
  static constexpr std::string_view property_name{
      "sycl::ext::oneapi::experimental::work_group_size_property"};

  static_assert(
      sizeof...(Dims) >= 1 && sizeof...(Dims) <= 3,
      "work_group_size property currently only supports up to three values.");
  static_assert((((Dims != 0) && ...)),
                "work_group_size property must only contain non-zero values.");

  constexpr size_t operator[](int Dim) const {
    return std::array{Dims...}[Dim];
  }

  static constexpr const char *ir_attribute_name = "sycl-work-group-size";
  static constexpr const char *ir_attribute_value = SizeListToStr<Dims...>::value;
};
template <size_t... Dims>
inline constexpr work_group_size_property<Dims...> work_group_size;


template <size_t... Dims>
struct work_group_size_hint_property
    : new_properties::detail::property_base<
          work_group_size_hint_property<Dims...>,
          struct work_group_size_hint_key> {
  static constexpr std::string_view property_name{
      "sycl::ext::oneapi::experimental::work_group_size_hint_property"};

  static_assert(sizeof...(Dims) >= 1 && sizeof...(Dims) <= 3,
                "work_group_size_hint property currently only supports up to "
                "three values.");
  static_assert(
      (((Dims != 0) && ...)),
      "work_group_size_hint property must only contain non-zero values.");

  constexpr size_t operator[](int Dim) const {
    return std::array{Dims...}[Dim];
  }

  static constexpr const char *ir_attribute_name = "sycl-work-group-size-hint";
  static constexpr const char *ir_attribute_value =
      SizeListToStr<Dims...>::value;
};
template <size_t... Dims>
inline constexpr work_group_size_hint_property<Dims...> work_group_size_hint;

template <uint32_t Size>
struct sub_group_size_property
    : new_properties::detail::property_base<sub_group_size_property<Size>,
                                            struct sub_group_size_key> {
  static constexpr std::string_view property_name{
      "sycl::ext::oneapi::experimental::sub_group_size_property"};

  static_assert(Size != 0,
                "sub_group_size_key property must contain a non-zero value.");

  // TODO: Is this still needed?
  using value_t = std::integral_constant<uint32_t, Size>;

  static constexpr uint32_t value = Size;

  static constexpr const char *ir_attribute_name = "sycl-sub-group-size";
  static constexpr uint32_t ir_attribute_value = Size;
};
template <uint32_t Size>
inline constexpr sub_group_size_property<Size> sub_group_size;

template <aspect... Aspects>
struct device_has_property
    : new_properties::detail::property_base<device_has_property<Aspects...>,
                                            struct device_has_key> {
  static constexpr std::string_view property_name{
      "sycl::ext::oneapi::experimental::device_has_property"};

  static constexpr std::array<aspect, sizeof...(Aspects)> value{Aspects...};

  static constexpr const char *ir_attribute_name = "sycl-device-has";
  static constexpr const char *ir_attribute_value =
      SizeListToStr<static_cast<size_t>(Aspects)...>::value;
};
template <aspect... Aspects>
inline constexpr device_has_property<Aspects...> device_has;

template <int Dims>
struct nd_range_kernel_property
    : new_properties::detail::property_base<nd_range_kernel_property<Dims>,
                                            struct nd_range_kernel_key> {
  static constexpr std::string_view property_name{
      "sycl::ext::oneapi::experimental::nd_range_kernel_property"};

  static_assert(
      Dims >= 1 && Dims <= 3,
      "nd_range_kernel_key property must use dimension of 1, 2 or 3.");

  static constexpr int dimensions = Dims;

  static constexpr const char *ir_attribute_name = "sycl-nd-range-kernel";
  static constexpr int ir_attribute_value = Dims;
};
template <int Dims>
inline constexpr nd_range_kernel_property<Dims> nd_range_kernel;

  // TODO: Add single_task_kernel_key for compatibility?
struct single_task_kernel_property
    : new_properties::detail::property_base<single_task_kernel_property> {
  static constexpr const char *ir_attribute_name = "sycl-single-task-kernel";
  static constexpr int ir_attribute_value = 0;
};
inline constexpr single_task_kernel_property single_task_kernel;

template <size_t... Dims>
struct max_work_group_size_property : new_properties::detail::property_base<
                                          max_work_group_size_property<Dims...>,
                                          struct max_work_group_size_key> {
  static constexpr std::string_view property_name{
      "sycl::ext::oneapi::experimental::max_work_group_size_property"};

  static_assert(
      sizeof...(Dims) >= 1 && sizeof...(Dims) <= 3,
      "max_work_group_size property currently only supports up to three values.");
  static_assert((((Dims != 0) && ...)),
                "max_work_group_size property must only contain non-zero values.");

  constexpr size_t operator[](int Dim) const {
    return std::array{Dims...}[Dim];
  }

  static constexpr const char *ir_attribute_name = "sycl-max-work-group-size";
  static constexpr const char *ir_attribute_value = SizeListToStr<Dims...>::value;
};
template <size_t... Dims>
inline constexpr max_work_group_size_property<Dims...> max_work_group_size;

template <size_t Size>
struct max_linear_work_group_size_property
    : new_properties::detail::property_base<
          max_linear_work_group_size_property<Size>,
          struct max_linear_work_group_size_key> {
  static constexpr std::string_view property_name{
      "sycl::ext::oneapi::experimental::max_linear_work_group_size_property"};

  static constexpr const char *ir_attribute_name = "sycl-max-linear-work-group-size";
  static constexpr size_t ir_attribute_value = Size;
};
template <size_t Size>
inline constexpr max_linear_work_group_size_property<Size>
    max_linear_work_group_size;

template <forward_progress_guarantee Guarantee,
          execution_scope CoordinationScope>
struct work_group_progress_property
    : new_properties::detail::property_base<
          work_group_progress_property<Guarantee, CoordinationScope>,
          struct work_group_progress_key> {
  static constexpr std::string_view property_name{
      "sycl::ext::oneapi::experimental::work_group_progress_property"};

  static constexpr forward_progress_guarantee guarantee = Guarantee;
  static constexpr execution_scope coordinationScope = CoordinationScope;
};
template <forward_progress_guarantee Guarantee,
          execution_scope CoordinationScope>
inline constexpr work_group_progress_property<Guarantee, CoordinationScope>
    work_group_progress;

template <forward_progress_guarantee Guarantee,
          execution_scope CoordinationScope>
struct sub_group_progress_property
    : new_properties::detail::property_base<
          sub_group_progress_property<Guarantee, CoordinationScope>,
          struct sub_group_progress_key> {
  static constexpr std::string_view property_name{
      "sycl::ext::oneapi::experimental::sub_group_progress_property"};

  static constexpr forward_progress_guarantee guarantee = Guarantee;
  static constexpr execution_scope coordinationScope = CoordinationScope;
};
template <forward_progress_guarantee Guarantee,
          execution_scope CoordinationScope>
inline constexpr sub_group_progress_property<Guarantee, CoordinationScope>
    sub_group_progress;

template <forward_progress_guarantee Guarantee,
          execution_scope CoordinationScope>
struct work_item_progress_property
    : new_properties::detail::property_base<
          work_item_progress_property<Guarantee, CoordinationScope>,
          struct work_item_progress_key> {
  static constexpr std::string_view property_name{
      "sycl::ext::oneapi::experimental::work_item_progress_property"};

  static constexpr forward_progress_guarantee guarantee = Guarantee;
  static constexpr execution_scope coordinationScope = CoordinationScope;
};
template <forward_progress_guarantee Guarantee,
          execution_scope CoordinationScope>
inline constexpr work_item_progress_property<Guarantee, CoordinationScope>
    work_item_progress;

namespace detail {

template <size_t... Dims>
struct HasCompileTimeEffect<work_group_size_key::value_t<Dims...>>
    : std::true_type {};
template <size_t... Dims>
struct HasCompileTimeEffect<work_group_size_hint_key::value_t<Dims...>>
    : std::true_type {};
template <uint32_t Size>
struct HasCompileTimeEffect<sub_group_size_key::value_t<Size>>
    : std::true_type {};
template <sycl::aspect... Aspects>
struct HasCompileTimeEffect<device_has_key::value_t<Aspects...>>
    : std::true_type {};

template <typename T, typename = void>
struct HasKernelPropertiesGetMethod : std::false_type {};

template <typename T>
struct HasKernelPropertiesGetMethod<T,
                                    std::void_t<decltype(std::declval<T>().get(
                                        std::declval<properties_tag>()))>>
    : std::true_type {
  using properties_t =
      decltype(std::declval<T>().get(std::declval<properties_tag>()));
};

// Trait for property compile-time meta names and values.
template <typename PropertyT> struct WGSizePropertyMetaInfo {
  static constexpr std::array<size_t, 0> WGSize = {};
  static constexpr size_t LinearSize = 0;
};

template <size_t Dim0, size_t... Dims>
struct WGSizePropertyMetaInfo<work_group_size_key::value_t<Dim0, Dims...>> {
  static constexpr std::array<size_t, sizeof...(Dims) + 1> WGSize = {Dim0,
                                                                     Dims...};
  static constexpr size_t LinearSize = (Dim0 * ... * Dims);
};

template <size_t Dim0, size_t... Dims>
struct WGSizePropertyMetaInfo<max_work_group_size_key::value_t<Dim0, Dims...>> {
  static constexpr std::array<size_t, sizeof...(Dims) + 1> WGSize = {Dim0,
                                                                     Dims...};
  static constexpr size_t LinearSize = (Dim0 * ... * Dims);
};

// Get the value of a work-group size related property from a property list
template <typename PropKey, typename PropertiesT>
struct GetWGPropertyFromPropList {};

template <typename PropKey, typename... PropertiesT>
struct GetWGPropertyFromPropList<PropKey, std::tuple<PropertiesT...>> {
  using prop_val_t = std::conditional_t<
      ContainsProperty<PropKey, std::tuple<PropertiesT...>>::value,
      typename FindCompileTimePropertyValueType<
          PropKey, std::tuple<PropertiesT...>>::type,
      void>;
  static constexpr auto WGSize =
      WGSizePropertyMetaInfo<std::remove_const_t<prop_val_t>>::WGSize;
  static constexpr size_t LinearSize =
      WGSizePropertyMetaInfo<std::remove_const_t<prop_val_t>>::LinearSize;
};

// If work_group_size and max_work_group_size coexist, check that the
// dimensionality matches and that the required work-group size doesn't
// trivially exceed the maximum size.
template <typename Properties>
struct ConflictingProperties<max_work_group_size_key, Properties>
    : std::false_type {
  using WGSizeVal = GetWGPropertyFromPropList<work_group_size_key, Properties>;
  using MaxWGSizeVal =
      GetWGPropertyFromPropList<max_work_group_size_key, Properties>;
  // If work_group_size_key doesn't exist in the list of properties, WGSize is
  // an empty array and so Dims == 0.
  static constexpr size_t Dims = WGSizeVal::WGSize.size();
  static_assert(
      Dims == 0 || Dims == MaxWGSizeVal::WGSize.size(),
      "work_group_size and max_work_group_size dimensionality must match");
  static_assert(Dims < 1 || WGSizeVal::WGSize[0] <= MaxWGSizeVal::WGSize[0],
                "work_group_size must not exceed max_work_group_size");
  static_assert(Dims < 2 || WGSizeVal::WGSize[1] <= MaxWGSizeVal::WGSize[1],
                "work_group_size must not exceed max_work_group_size");
  static_assert(Dims < 3 || WGSizeVal::WGSize[2] <= MaxWGSizeVal::WGSize[2],
                "work_group_size must not exceed max_work_group_size");
};

// If work_group_size and max_linear_work_group_size coexist, check that the
// required linear work-group size doesn't trivially exceed the maximum size.
template <typename Properties>
struct ConflictingProperties<max_linear_work_group_size_key, Properties>
    : std::false_type {
  using WGSizeVal = GetWGPropertyFromPropList<work_group_size_key, Properties>;
  using MaxLinearWGSizeVal =
      GetPropertyValueFromPropList<max_linear_work_group_size_key, size_t, void,
                                   Properties>;
  static_assert(WGSizeVal::WGSize.empty() ||
                    WGSizeVal::LinearSize <= MaxLinearWGSizeVal::value,
                "work_group_size must not exceed max_linear_work_group_size");
};

} // namespace detail
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl

#ifdef __SYCL_DEVICE_ONLY__
#define SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(PROP)                                \
  [[__sycl_detail__::add_ir_attributes_function(                               \
      sycl::ext::oneapi::experimental::detail::PropertyMetaInfo<               \
          std::remove_cv_t<std::remove_reference_t<decltype(PROP)>>>::name,    \
      sycl::ext::oneapi::experimental::detail::PropertyMetaInfo<               \
          std::remove_cv_t<std::remove_reference_t<decltype(PROP)>>>::value)]]
#else
#define SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(PROP)
#endif
