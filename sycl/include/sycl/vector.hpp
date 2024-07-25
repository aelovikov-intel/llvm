//==---------------- vector.hpp --- Implements sycl::vec -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// Check if Clang's ext_vector_type attribute is available. Host compiler
// may not be Clang, and Clang may not be built with the extension.
#ifdef __clang__
#ifndef __has_extension
#define __has_extension(x) 0
#endif
#ifdef __HAS_EXT_VECTOR_TYPE__
#error "Undefine __HAS_EXT_VECTOR_TYPE__ macro"
#endif
#if __has_extension(attribute_ext_vector_type)
#define __HAS_EXT_VECTOR_TYPE__
#endif
#endif // __clang__

#if !defined(__HAS_EXT_VECTOR_TYPE__) && defined(__SYCL_DEVICE_ONLY__)
#error "SYCL device compiler is built without ext_vector_type support"
#endif

#include <sycl/access/access.hpp>              // for decorated, address_space
#include <sycl/aliases.hpp>                    // for half, cl_char, cl_int
#include <sycl/detail/common.hpp>              // for ArrayCreator, RepeatV...
#include <sycl/detail/defines_elementary.hpp>  // for __SYCL2020_DEPRECATED
#include <sycl/detail/generic_type_lists.hpp>  // for vector_basic_list
#include <sycl/detail/generic_type_traits.hpp> // for is_sigeninteger, is_s...
#include <sycl/detail/memcpy.hpp>              // for memcpy
#include <sycl/detail/named_swizzles_mixin.hpp>
#include <sycl/detail/type_list.hpp>   // for is_contained
#include <sycl/detail/type_traits.hpp> // for is_floating_point
#include <sycl/detail/vec_operators_mixins.hpp>
#include <sycl/detail/vector_arith.hpp>
#include <sycl/detail/vector_convert.hpp> // for convertImpl
#include <sycl/detail/vector_traits.hpp>  // for vector_alignment
#include <sycl/half_type.hpp>             // for StorageT, half, Vec16...

#include <sycl/ext/oneapi/bfloat16.hpp> // bfloat16

#include <algorithm>   // for std::min
#include <array>       // for array
#include <cassert>     // for assert
#include <cstddef>     // for size_t, NULL, byte
#include <cstdint>     // for uint8_t, int16_t, int...
#include <functional>  // for divides, multiplies
#include <iterator>    // for pair
#include <ostream>     // for operator<<, basic_ost...
#include <type_traits> // for enable_if_t, is_same
#include <utility>     // for index_sequence, make_...

namespace sycl {
inline namespace _V1 {

struct elem {
  static constexpr int x = 0;
  static constexpr int y = 1;
  static constexpr int z = 2;
  static constexpr int w = 3;
  static constexpr int r = 0;
  static constexpr int g = 1;
  static constexpr int b = 2;
  static constexpr int a = 3;
  static constexpr int s0 = 0;
  static constexpr int s1 = 1;
  static constexpr int s2 = 2;
  static constexpr int s3 = 3;
  static constexpr int s4 = 4;
  static constexpr int s5 = 5;
  static constexpr int s6 = 6;
  static constexpr int s7 = 7;
  static constexpr int s8 = 8;
  static constexpr int s9 = 9;
  static constexpr int sA = 10;
  static constexpr int sB = 11;
  static constexpr int sC = 12;
  static constexpr int sD = 13;
  static constexpr int sE = 14;
  static constexpr int sF = 15;
};

namespace detail {
// Templated vs. non-templated conversion operator behaves differently when two
// conversions are needed as in the case below:
//
//   sycl::vec<int, 1> v;
//   std::ignore = static_cast<bool>(v);
//
// Make sure the snippet above compiles. That is important because
//
//   sycl::vec<int, 2> v;
//   if (v.x() == 42)
//     ...
//
// must go throw `v.x()` returning a swizzle, then its `operator==` returning
// vec<int, 1> and we want that code to compile.
template <typename Self, typename T, int N, typename = void>
struct ScalarConversionOperatorMixIn {};

template <typename Self, typename T, int N>
struct ScalarConversionOperatorMixIn<Self, T, N, std::enable_if_t<N == 1>> {
  operator T() const { return (*static_cast<const Self *>(this))[0]; }
};

template <typename VecT, int... Indexes>
class Swizzle;

template <typename VecT, int... Indexes>
inline constexpr bool is_assignable_swizzle = !std::is_const_v<VecT>; // FIXME:

template <typename Self, typename DataT, int N, bool IsAssignable>
struct SwizzleBase {
  const Self &operator=(const Self &) = delete;
};
template <typename Self, typename DataT, int N>
struct SwizzleBase<Self, DataT, N, true> {
  template <access::address_space AddressSpace, access::decorated IsDecorated>
  void load(size_t offset,
            multi_ptr<const DataT, AddressSpace, IsDecorated> ptr) const {
    vec<DataT, N> v;
    v.load(offset, ptr);
    *static_cast<Self *>(this) = v;
  }

  template <typename OtherVecT, int... OtherIndexes>
  std::enable_if_t<std::is_same_v<typename OtherVecT::element_type, DataT> &&
                       sizeof...(OtherIndexes) == N,
                   const Self &>
  operator=(const Swizzle<OtherVecT, OtherIndexes...> &rhs) {
    return (*this = static_cast<vec<DataT, N>>(rhs));
  }

  const Self &operator=(const vec<DataT, N> &rhs) const {
    for (int i = 0; i < N; ++i)
      (*static_cast<const Self *>(this))[i] = rhs[i];

    return *static_cast<const Self *>(this);
  }

  const Self &operator=(const DataT &rhs) const {
    for (int i = 0; i < N; ++i)
      (*static_cast<const Self *>(this))[i] = rhs;

    return *static_cast<const Self *>(this);
  }
};

template <typename VecT, int... Indexes>
class Swizzle
    : public ScalarConversionOperatorMixIn<Swizzle<VecT, Indexes...>,
                                           typename VecT::element_type,
                                           sizeof...(Indexes)>,
      public SwizzleBase<Swizzle<VecT, Indexes...>, typename VecT::element_type,
                         sizeof...(Indexes),
                         is_assignable_swizzle<VecT, Indexes...>>,
      public std::conditional_t<is_assignable_swizzle<VecT, Indexes...>,
                                SwizzleOpAssignMixins<Swizzle<VecT, Indexes...>,
                                                      VecT, sizeof...(Indexes)>,
                                SwizzleBaseMixins<Swizzle<VecT, Indexes...>,
                                                  VecT, sizeof...(Indexes)>>,
      public std::conditional_t<
          is_assignable_swizzle<VecT, Indexes...>,
          PrefixPostfixIncDecMixin<Swizzle<VecT, Indexes...>,
                                   typename VecT::element_type>,
          PrefixIncDecMixin<Swizzle<VecT, Indexes...>,
                            typename VecT::element_type>> {
  using DataT = typename VecT::element_type;
  static constexpr int NumElements = sizeof...(Indexes);
  using ResultVec = vec<DataT, NumElements>;

  // Get underlying vec index for (*this)[idx] access.
  static constexpr auto get_vec_idx(int idx) {
    int counter = 0;
    int result = -1;
    ((result = counter++ == idx ? Indexes : result), ...);
    return result;
  }

public:
  using SwizzleBase<Swizzle<VecT, Indexes...>, typename VecT::element_type,
                    sizeof...(Indexes),
                    is_assignable_swizzle<VecT, Indexes...>>::operator=;

  using element_type = DataT;
  using value_type = DataT;

#ifdef __SYCL_DEVICE_ONLY__
  using vector_t = typename ResultVec::vector_t;
#endif

  Swizzle() = delete;
  Swizzle(const Swizzle &) = delete;

  Swizzle(VecT &Vec) : Vec(Vec) {}

#ifdef __SYCL_DEVICE_ONLY__
  operator vector_t() const {
    return static_cast<vector_t>(static_cast<ResultVec>(*this));
  }
#endif

  static constexpr size_t byte_size() noexcept {
    return ResultVec::byte_size();
  }
  static constexpr size_t size() noexcept { return ResultVec::size(); }

  __SYCL2020_DEPRECATED(
      "get_size() is deprecated, please use byte_size() instead")
  size_t get_size() const { return static_cast<ResultVec>(*this).get_size(); }

  __SYCL2020_DEPRECATED("get_count() is deprecated, please use size() instead")
  size_t get_count() const {
    return static_cast<ResultVec>(*this).get_count();
  };

  template <typename ConvertT,
            rounding_mode RoundingMode = rounding_mode::automatic>
  vec<ConvertT, NumElements> convert() const {
    return static_cast<ResultVec>(*this)
        .template convert<ConvertT, RoundingMode>();
  }

  template <typename asT> asT as() const {
    return static_cast<ResultVec>(*this).template as<asT>();
  }

  template <access::address_space AddressSpace, access::decorated IsDecorated>
  void store(size_t offset,
             multi_ptr<DataT, AddressSpace, IsDecorated> ptr) const {
    return static_cast<ResultVec>(*this).store(offset, ptr);
  }

  operator ResultVec() const { return ResultVec{Vec[Indexes]...}; }

  template <int... swizzleIndexes> auto swizzle() const {
    return Vec.template swizzle<get_vec_idx(swizzleIndexes)...>();
  }

  auto &operator[](int index) const { return Vec[get_vec_idx(index)]; }

public:
  VecT &Vec;
};
} // namespace detail

///////////////////////// class sycl::vec /////////////////////////
// Provides a cross-platform vector class template that works efficiently on
// SYCL devices as well as in host C++ code.
template <typename DataT, int NumElements>
class __SYCL_EBO vec
    : public detail::vec_arith<DataT, NumElements>,
      public detail::ScalarConversionOperatorMixIn<vec<DataT, NumElements>,
                                                   DataT, NumElements>,
      public detail::NamedSwizzlesMixInBoth<vec<DataT, NumElements>,
                                            NumElements> {

  static_assert(NumElements == 1 || NumElements == 2 || NumElements == 3 ||
                    NumElements == 4 || NumElements == 8 || NumElements == 16,
                "Invalid number of elements for sycl::vec: only 1, 2, 3, 4, 8 "
                "or 16 are supported");
  static_assert(sizeof(bool) == sizeof(uint8_t), "bool size is not 1 byte");

  // https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#memory-layout-and-alignment
  // It is required by the SPEC to align vec<DataT, 3> with vec<DataT, 4>.
  static constexpr size_t AdjustedNum = (NumElements == 3) ? 4 : NumElements;

  // This represent type of underlying value. There should be only one field
  // in the class, so vec<float, 16> should be equal to float16 in memory.
  using DataType = std::array<DataT, AdjustedNum>;

#ifdef __SYCL_DEVICE_ONLY__
  using element_type_for_vector_t = typename detail::map_type<
      DataT,
#if (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)
      std::byte, /*->*/ std::uint8_t, //
#endif
      bool, /*->*/ std::uint8_t,                            //
      sycl::half, /*->*/ sycl::detail::half_impl::StorageT, //
      sycl::ext::oneapi::bfloat16,
      /*->*/ sycl::ext::oneapi::detail::Bfloat16StorageT, //
      char, /*->*/ detail::ConvertToOpenCLType_t<char>,   //
      DataT, /*->*/ DataT                                 //
      >::type;

public:
  // Type used for passing sycl::vec to SPIRV builtins.
  // We can not use ext_vector_type(1) as it's not supported by SPIRV
  // plugins (CTS fails).
  using vector_t =
      typename std::conditional_t<NumElements == 1, element_type_for_vector_t,
                                  element_type_for_vector_t __attribute__((
                                      ext_vector_type(NumElements)))>;

private:
#endif // __SYCL_DEVICE_ONLY__

  static constexpr int getNumElements() { return NumElements; }

  // SizeChecker is needed for vec(const argTN &... args) ctor to validate args.
  template <int Counter, int MaxValue, class...>
  struct SizeChecker : std::conditional_t<Counter == MaxValue, std::true_type,
                                          std::false_type> {};

  template <int Counter, int MaxValue, typename DataT_, class... tail>
  struct SizeChecker<Counter, MaxValue, DataT_, tail...>
      : std::conditional_t<Counter + 1 <= MaxValue,
                           SizeChecker<Counter + 1, MaxValue, tail...>,
                           std::false_type> {};

  // Utility trait for creating an std::array from an vector argument.
  template <typename DataT_, typename T, std::size_t... Is>
  static constexpr std::array<DataT_, sizeof...(Is)>
  VecToArray(const vec<T, sizeof...(Is)> &V, std::index_sequence<Is...>) {
    return {static_cast<DataT_>(V[Is])...};
  }
  template <typename DataT_, typename T, int N, int... T5, std::size_t... Is>
  static constexpr std::array<DataT_, sizeof...(Is)>
  VecToArray(const detail::Swizzle<vec<T, N>, T5...> &V,
             std::index_sequence<Is...>) {
    return {static_cast<DataT_>(V[Is])...};
  }
  template <typename DataT_, typename T, int N, int... T5, std::size_t... Is>
  static constexpr std::array<DataT_, sizeof...(Is)>
  VecToArray(const detail::Swizzle<const vec<T, N>, T5...> &V,
             std::index_sequence<Is...>) {
    return {static_cast<DataT_>(V[Is])...};
  }

  template <typename DataT_, typename T, int N>
  static constexpr std::array<DataT_, N>
  FlattenVecArgHelper(const vec<T, N> &A) {
    return VecToArray<DataT_>(A, std::make_index_sequence<N>());
  }
  template <typename DataT_, typename T, int N, int... Indexes_>
  static constexpr std::array<DataT_, sizeof...(Indexes_)> FlattenVecArgHelper(
      const detail::Swizzle<vec<T, N>, Indexes_...> &A) {
    return VecToArray<DataT_>(A,
                              std::make_index_sequence<sizeof...(Indexes_)>());
  }
  template <typename DataT_, typename T, int N, int... Indexes_>
  static constexpr std::array<DataT_, sizeof...(Indexes_)> FlattenVecArgHelper(
      const detail::Swizzle<const vec<T, N>, Indexes_...> &A) {
    return VecToArray<DataT_>(A,
                              std::make_index_sequence<sizeof...(Indexes_)>());
  }
  template <typename DataT_, typename T>
  static constexpr auto FlattenVecArgHelper(const T &A) {
    // static_cast required to avoid narrowing conversion warning
    // when T = unsigned long int and DataT_ = int.
    return std::array<DataT_, 1>{static_cast<DataT_>(A)};
  }
  template <typename DataT_, typename T> struct FlattenVecArg {
    constexpr auto operator()(const T &A) const {
      return FlattenVecArgHelper<DataT_>(A);
    }
  };

  // Alias for shortening the vec arguments to array converter.
  template <typename DataT_, typename... ArgTN>
  using VecArgArrayCreator =
      detail::ArrayCreator<DataT_, FlattenVecArg, ArgTN...>;

#define __SYCL_ALLOW_VECTOR_SIZES(num_elements)                                \
  template <int Counter, int MaxValue, typename DataT_, class... tail>         \
  struct SizeChecker<Counter, MaxValue, vec<DataT_, num_elements>, tail...>    \
      : std::conditional_t<                                                    \
            Counter + (num_elements) <= MaxValue,                              \
            SizeChecker<Counter + (num_elements), MaxValue, tail...>,          \
            std::false_type> {};                                               \
  template <int Counter, int MaxValue, typename DataT_, int... Indexes_,       \
            class... tail>                                                     \
  struct SizeChecker<Counter, MaxValue,                                        \
                     detail::Swizzle<vec<DataT_, num_elements>, Indexes_...>,  \
                     tail...>                                                  \
      : std::conditional_t<                                                    \
            Counter + sizeof...(Indexes_) <= MaxValue,                         \
            SizeChecker<Counter + sizeof...(Indexes_), MaxValue, tail...>,     \
            std::false_type> {};                                               \
  template <int Counter, int MaxValue, typename DataT_, int... Indexes_,       \
            class... tail>                                                     \
  struct SizeChecker<                                                          \
      Counter, MaxValue,                                                       \
      detail::Swizzle<const vec<DataT_, num_elements>, Indexes_...>, tail...>  \
      : std::conditional_t<                                                    \
            Counter + sizeof...(Indexes_) <= MaxValue,                         \
            SizeChecker<Counter + sizeof...(Indexes_), MaxValue, tail...>,     \
            std::false_type> {};

  __SYCL_ALLOW_VECTOR_SIZES(1)
  __SYCL_ALLOW_VECTOR_SIZES(2)
  __SYCL_ALLOW_VECTOR_SIZES(3)
  __SYCL_ALLOW_VECTOR_SIZES(4)
  __SYCL_ALLOW_VECTOR_SIZES(8)
  __SYCL_ALLOW_VECTOR_SIZES(16)
#undef __SYCL_ALLOW_VECTOR_SIZES

  // TypeChecker is needed for vec(const argTN &... args) ctor to validate args.
  template <typename T, typename DataT_>
  struct TypeChecker : std::is_convertible<T, DataT_> {};
#define __SYCL_ALLOW_VECTOR_TYPES(num_elements)                                \
  template <typename DataT_>                                                   \
  struct TypeChecker<vec<DataT_, num_elements>, DataT_> : std::true_type {};   \
  template <typename DataT_, int... Indexes_>                                  \
  struct TypeChecker<detail::Swizzle<vec<DataT_, num_elements>, Indexes_...>,  \
                     DataT_> : std::true_type {};                              \
  template <typename DataT_, int... Indexes_>                                  \
  struct TypeChecker<                                                          \
      detail::Swizzle<const vec<DataT_, num_elements>, Indexes_...>, DataT_>   \
      : std::true_type {};

  __SYCL_ALLOW_VECTOR_TYPES(1)
  __SYCL_ALLOW_VECTOR_TYPES(2)
  __SYCL_ALLOW_VECTOR_TYPES(3)
  __SYCL_ALLOW_VECTOR_TYPES(4)
  __SYCL_ALLOW_VECTOR_TYPES(8)
  __SYCL_ALLOW_VECTOR_TYPES(16)
#undef __SYCL_ALLOW_VECTOR_TYPES

  template <int... Indexes>
  using Swizzle = detail::Swizzle<vec, Indexes...>;

  template <int... Indexes>
  using ConstSwizzle = detail::Swizzle<const vec, Indexes...>;

  // Shortcuts for args validation in vec(const argTN &... args) ctor.
  template <typename... argTN>
  using EnableIfSuitableTypes = typename std::enable_if_t<
      std::conjunction_v<TypeChecker<argTN, DataT>...>>;

  template <typename... argTN>
  using EnableIfSuitableNumElements =
      typename std::enable_if_t<SizeChecker<0, NumElements, argTN...>::value>;

  // Element type for relational operator return value.
  using rel_t = detail::select_cl_scalar_integral_signed_t<DataT>;

public:
  // Aliases required by SYCL 2020 to make sycl::vec consistent
  // with that of marray and buffer.
  using element_type = DataT;
  using value_type = DataT;

  /****************** Constructors **************/
  vec() = default;
  constexpr vec(const vec &Rhs) = default;
  constexpr vec(vec &&Rhs) = default;

private:
  // Implementation detail for the next public ctor.
  template <size_t... Is>
  constexpr vec(const std::array<DataT, NumElements> &Arr,
                std::index_sequence<Is...>)
      : m_Data{Arr[Is]...} {}

public:
  explicit constexpr vec(const DataT &arg)
      : vec{detail::RepeatValue<NumElements>(arg),
            std::make_index_sequence<NumElements>()} {}

  // Constructor from values of base type or vec of base type. Checks that
  // base types are match and that the NumElements == sum of lengths of args.
  template <typename... argTN, typename = EnableIfSuitableTypes<argTN...>,
            typename = EnableIfSuitableNumElements<argTN...>>
  constexpr vec(const argTN &...args)
      : vec{VecArgArrayCreator<DataT, argTN...>::Create(args...),
            std::make_index_sequence<NumElements>()} {}

  /****************** Assignment Operators **************/
  constexpr vec &operator=(const vec &Rhs) = default;

  // Template required to prevent ambiguous overload with the copy assignment
  // when NumElements == 1. The template prevents implicit conversion from
  // vec<_, 1> to DataT.
  template <typename Ty = DataT>
  typename std::enable_if_t<
      std::is_fundamental_v<Ty> ||
          detail::is_half_or_bf16_v<typename std::remove_const_t<Ty>>,
      vec &>
  operator=(const DataT &Rhs) {
    *this = vec{Rhs};
    return *this;
  }

  // W/o this, things like "vec<char,*> = vec<signed char, *>" doesn't work.
  template <typename Ty = DataT>
  typename std::enable_if_t<
      !std::is_same_v<Ty, rel_t> && std::is_convertible_v<Ty, rel_t>, vec &>
  operator=(const vec<rel_t, NumElements> &Rhs) {
    *this = Rhs.template as<vec>();
    return *this;
  }

#ifdef __SYCL_DEVICE_ONLY__
  // Make it a template to avoid ambiguity with `vec(const DataT &)` when
  // `vector_t` is the same as `DataT`. Not that the other ctor isn't a template
  // so we don't even need a smart `enable_if` condition here, the mere fact of
  // this being a template makes the other ctor preferred.
  template <
      typename vector_t_ = vector_t,
      typename = typename std::enable_if_t<std::is_same_v<vector_t_, vector_t>>>
  constexpr vec(vector_t_ openclVector) {
    m_Data = sycl::bit_cast<DataType>(openclVector);
  }

  /* @SYCL2020
   * Available only when: compiled for the device.
   * Converts this SYCL vec instance to the underlying backend-native vector
   * type defined by vector_t.
   */
  operator vector_t() const { return sycl::bit_cast<vector_t>(m_Data); }
#endif // __SYCL_DEVICE_ONLY__

  __SYCL2020_DEPRECATED("get_count() is deprecated, please use size() instead")
  static constexpr size_t get_count() { return size(); }
  static constexpr size_t size() noexcept { return NumElements; }
  __SYCL2020_DEPRECATED(
      "get_size() is deprecated, please use byte_size() instead")
  static constexpr size_t get_size() { return byte_size(); }
  static constexpr size_t byte_size() noexcept { return sizeof(m_Data); }

private:
  // We interpret bool as int8_t, std::byte as uint8_t for conversion to other
  // types.
  template <typename T>
  using ConvertBoolAndByteT =
      typename detail::map_type<T,
#if (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)
                                std::byte, /*->*/ std::uint8_t, //
#endif
                                bool, /*->*/ std::uint8_t, //
                                T, /*->*/ T                //
                                >::type;

  // getValue should be able to operate on different underlying
  // types: enum cl_float#N , builtin vector float#N, builtin type float.
  constexpr auto getValue(int Index) const {
    using RetType =
        typename std::conditional_t<detail::is_byte_v<DataT>, int8_t,
#ifdef __SYCL_DEVICE_ONLY__
                                    element_type_for_vector_t
#else
                                    DataT
#endif
                                    >;

#ifdef __SYCL_DEVICE_ONLY__
    if constexpr (std::is_same_v<DataT, sycl::ext::oneapi::bfloat16>)
      return sycl::bit_cast<RetType>(m_Data[Index]);
    else
#endif
      return static_cast<RetType>(m_Data[Index]);
  }

public:
  template <typename convertT,
            rounding_mode roundingMode = rounding_mode::automatic>
  vec<convertT, NumElements> convert() const {

    using T = ConvertBoolAndByteT<DataT>;
    using R = ConvertBoolAndByteT<convertT>;
    using bfloat16 = sycl::ext::oneapi::bfloat16;
    static_assert(std::is_integral_v<R> ||
                      detail::is_floating_point<R>::value ||
                      std::is_same_v<R, bfloat16>,
                  "Unsupported convertT");

    using OpenCLT = detail::ConvertToOpenCLType_t<T>;
    using OpenCLR = detail::ConvertToOpenCLType_t<R>;
    vec<convertT, NumElements> Result;

    // convertImpl can't be called with the same From and To types and therefore
    // we need some special processing in a few cases.
    if constexpr (std::is_same_v<DataT, convertT>) {
      return *this;
    } else if constexpr (std::is_same_v<OpenCLT, OpenCLR> ||
                         std::is_same_v<T, R>) {
      for (size_t I = 0; I < NumElements; ++I)
        Result[I] = static_cast<convertT>(getValue(I));
      return Result;
    } else {

#ifdef __SYCL_DEVICE_ONLY__
      using OpenCLVecT = OpenCLT __attribute__((ext_vector_type(NumElements)));
      using OpenCLVecR = OpenCLR __attribute__((ext_vector_type(NumElements)));

      auto NativeVector = sycl::bit_cast<vector_t>(*this);
      using ConvertTVecType = typename vec<convertT, NumElements>::vector_t;

      // Whole vector conversion can only be done, if:
      constexpr bool canUseNativeVectorConvert =
#ifdef __NVPTX__
          //  TODO: Likely unnecessary as
          //  https://github.com/intel/llvm/issues/11840 has been closed
          //  already.
          false &&
#endif
          NumElements > 1 &&
          // - vec storage has an equivalent OpenCL native vector it is
          //   implicitly convertible to. There are some corner cases where it
          //   is not the case with char, long and long long types.
          std::is_convertible_v<vector_t, OpenCLVecT> &&
          std::is_convertible_v<ConvertTVecType, OpenCLVecR> &&
          // - it is not a signed to unsigned (or vice versa) conversion
          //   see comments within 'convertImpl' for more details;
          !detail::is_sint_to_from_uint<T, R>::value &&
          // - destination type is not bool. bool is stored as integer under the
          //   hood and therefore conversion to bool looks like conversion
          //   between two integer types. Since bit pattern for true and false
          //   is not defined, there is no guarantee that integer conversion
          //   yields right results here;
          !std::is_same_v<convertT, bool>;

      if constexpr (canUseNativeVectorConvert) {
        auto val = detail::convertImpl<T, R, roundingMode, NumElements, OpenCLVecT,
                                OpenCLVecR>(NativeVector);
        Result.m_Data = sycl::bit_cast<decltype(Result.m_Data)>(val);
      } else
#endif // __SYCL_DEVICE_ONLY__
      {
        // Otherwise, we fallback to per-element conversion:
        for (size_t I = 0; I < NumElements; ++I) {
          auto val =
              detail::convertImpl<T, R, roundingMode, 1, OpenCLT, OpenCLR>(
                  getValue(I));
#ifdef __SYCL_DEVICE_ONLY__
          // On device, we interpret BF16 as uint16.
          if constexpr (std::is_same_v<convertT, bfloat16>)
            Result[I] = sycl::bit_cast<convertT>(val);
          else
#endif
            Result[I] = static_cast<convertT>(val);
        }
      }
    }
    return Result;
  }

  template <typename asT> asT as() const { return sycl::bit_cast<asT>(*this); }

  template <int... SwizzleIndexes> Swizzle<SwizzleIndexes...> swizzle() {
    return *this;
  }

  template <int... SwizzleIndexes>
  ConstSwizzle<SwizzleIndexes...> swizzle() const {
    return *this;
  }

  const DataT &operator[](int i) const { return m_Data[i]; }

  DataT &operator[](int i) { return m_Data[i]; }

  template <access::address_space Space, access::decorated DecorateAddress>
  void load(size_t Offset, multi_ptr<const DataT, Space, DecorateAddress> Ptr) {
    for (int I = 0; I < NumElements; I++) {
      m_Data[I] = *multi_ptr<const DataT, Space, DecorateAddress>(
          Ptr + Offset * NumElements + I);
    }
  }
  template <access::address_space Space, access::decorated DecorateAddress>
  void load(size_t Offset, multi_ptr<DataT, Space, DecorateAddress> Ptr) {
    multi_ptr<const DataT, Space, DecorateAddress> ConstPtr(Ptr);
    load(Offset, ConstPtr);
  }
  template <int Dimensions, access::mode Mode,
            access::placeholder IsPlaceholder, access::target Target,
            typename PropertyListT>
  void
  load(size_t Offset,
       accessor<DataT, Dimensions, Mode, Target, IsPlaceholder, PropertyListT>
           Acc) {
    multi_ptr<const DataT, detail::TargetToAS<Target>::AS,
              access::decorated::yes>
        MultiPtr(Acc);
    load(Offset, MultiPtr);
  }
  void load(size_t Offset, const DataT *Ptr) {
    for (int I = 0; I < NumElements; ++I)
      m_Data[I] = Ptr[Offset * NumElements + I];
  }

  template <access::address_space Space, access::decorated DecorateAddress>
  void store(size_t Offset,
             multi_ptr<DataT, Space, DecorateAddress> Ptr) const {
    for (int I = 0; I < NumElements; I++) {
      *multi_ptr<DataT, Space, DecorateAddress>(Ptr + Offset * NumElements +
                                                I) = m_Data[I];
    }
  }
  template <int Dimensions, access::mode Mode,
            access::placeholder IsPlaceholder, access::target Target,
            typename PropertyListT>
  void
  store(size_t Offset,
        accessor<DataT, Dimensions, Mode, Target, IsPlaceholder, PropertyListT>
            Acc) {
    multi_ptr<DataT, detail::TargetToAS<Target>::AS, access::decorated::yes>
        MultiPtr(Acc);
    store(Offset, MultiPtr);
  }
  void store(size_t Offset, DataT *Ptr) const {
    for (int I = 0; I < NumElements; ++I)
      Ptr[Offset * NumElements + I] = m_Data[I];
  }

private:
  // fields
  // Alignment is the same as size, to a maximum size of 64. SPEC requires
  // "The elements of an instance of the SYCL vec class template are stored
  // in memory sequentially and contiguously and are aligned to the size of
  // the element type in bytes multiplied by the number of elements."
  static constexpr int alignment = (std::min)((size_t)64, sizeof(DataType));
  alignas(alignment) DataType m_Data;

  // friends
  template <typename T1, int T2> friend class __SYCL_EBO vec;
  // To allow arithmetic operators access private members of vec.
  template <typename T1, int T2> friend class detail::vec_arith;
  template <typename T1, int T2> friend class detail::vec_arith_common;
};
///////////////////////// class sycl::vec /////////////////////////

#ifdef __cpp_deduction_guides
// all compilers supporting deduction guides also support fold expressions
template <class T, class... U,
          class = std::enable_if_t<(std::is_same_v<T, U> && ...)>>
vec(T, U...) -> vec<T, sizeof...(U) + 1>;
#endif

} // namespace _V1
} // namespace sycl
