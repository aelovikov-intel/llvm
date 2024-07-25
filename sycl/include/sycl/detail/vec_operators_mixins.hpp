#pragma once

#include <functional>
#include <sycl/detail/generic_type_traits.hpp>
#include <sycl/detail/type_traits.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/half_type.hpp>
#include <type_traits>

namespace sycl {
inline namespace _V1 {

template <typename DataT, int NumElem> class __SYCL_EBO vec;

namespace detail {
struct ShiftLeft {
  template <class T, class U>
  constexpr auto operator()(T &&lhs, U &&rhs) const
      -> decltype(std::forward<T>(lhs) << std::forward<U>(rhs)) {
    return std::forward<T>(lhs) << std::forward<U>(rhs);
  }
};
struct ShiftRight {
  template <class T, class U>
  constexpr auto operator()(T &&lhs,
                            U &&rhs) const -> decltype(std::forward<T>(lhs) >>
                                                       std::forward<U>(rhs)) {
    return std::forward<T>(lhs) >> std::forward<U>(rhs);
  }
};

struct UnaryPlus {
  template <class T>
  constexpr auto operator()(T &&arg) const -> decltype(+std::forward<T>(arg)) {
    return +std::forward<T>(arg);
  }
};

template <class T>
static constexpr bool not_fp =
    !std::is_same_v<T, float> && !std::is_same_v<T, double> &&
    !std::is_same_v<T, half>;

template <typename Op, typename T>
inline constexpr bool is_op_available = false;

#define __SYCL_OP_AVAILABILITY(OP, COND)                                       \
  template <typename T> inline constexpr bool is_op_available<OP, T> = COND;

// clang-format off
__SYCL_OP_AVAILABILITY(std::plus<void>          , !detail::is_byte_v<T>)
__SYCL_OP_AVAILABILITY(std::minus<void>         , !detail::is_byte_v<T>)
__SYCL_OP_AVAILABILITY(std::multiplies<void>    , !detail::is_byte_v<T>)
__SYCL_OP_AVAILABILITY(std::divides<void>       , !detail::is_byte_v<T>)
__SYCL_OP_AVAILABILITY(std::modulus<void>       , !detail::is_byte_v<T> && not_fp<T>)

__SYCL_OP_AVAILABILITY(std::bit_and<void>       , not_fp<T>)
__SYCL_OP_AVAILABILITY(std::bit_or<void>        , not_fp<T>)
__SYCL_OP_AVAILABILITY(std::bit_xor<void>       , not_fp<T>)

__SYCL_OP_AVAILABILITY(std::equal_to<void>      , true)
__SYCL_OP_AVAILABILITY(std::not_equal_to<void>  , true)
__SYCL_OP_AVAILABILITY(std::less<void>          , !detail::is_byte_v<T>)
__SYCL_OP_AVAILABILITY(std::greater<void>       , !detail::is_byte_v<T>)
__SYCL_OP_AVAILABILITY(std::less_equal<void>    , !detail::is_byte_v<T>)
__SYCL_OP_AVAILABILITY(std::greater_equal<void> , !detail::is_byte_v<T>)

__SYCL_OP_AVAILABILITY(std::logical_and<void>   , !detail::is_byte_v<T>)
__SYCL_OP_AVAILABILITY(std::logical_or<void>    , !detail::is_byte_v<T>)

__SYCL_OP_AVAILABILITY(ShiftLeft     , !detail::is_byte_v<T> && not_fp<T>)
__SYCL_OP_AVAILABILITY(ShiftRight    , !detail::is_byte_v<T> && not_fp<T>)

// Unary
__SYCL_OP_AVAILABILITY(std::negate<void>       , !detail::is_byte_v<T>)
__SYCL_OP_AVAILABILITY(std::logical_not<void>  , !detail::is_byte_v<T>)
__SYCL_OP_AVAILABILITY(std::bit_not<void>      , not_fp<T>)
__SYCL_OP_AVAILABILITY(UnaryPlus               , !detail::is_byte_v<T>)
// clang-format on

#undef __SYCL_OP_AVAILABILITY

// clang-format off
#define __SYCL_PROCESS_BINARY_OPS(PROCESS_OP, DELIMITER) \
          PROCESS_OP(std::plus<void>)          \
DELIMITER PROCESS_OP(std::minus<void>)         \
DELIMITER PROCESS_OP(std::multiplies<void>)    \
DELIMITER PROCESS_OP(std::divides<void>)       \
DELIMITER PROCESS_OP(std::modulus<void>)       \
DELIMITER PROCESS_OP(std::bit_and<void>)       \
DELIMITER PROCESS_OP(std::bit_or<void>)        \
DELIMITER PROCESS_OP(std::bit_xor<void>)       \
DELIMITER PROCESS_OP(std::equal_to<void>)      \
DELIMITER PROCESS_OP(std::not_equal_to<void>)  \
DELIMITER PROCESS_OP(std::less<void>)          \
DELIMITER PROCESS_OP(std::greater<void>)       \
DELIMITER PROCESS_OP(std::less_equal<void>)    \
DELIMITER PROCESS_OP(std::greater_equal<void>) \
DELIMITER PROCESS_OP(std::logical_and<void>)   \
DELIMITER PROCESS_OP(std::logical_or<void>)    \
DELIMITER PROCESS_OP(ShiftLeft)                \
DELIMITER PROCESS_OP(ShiftRight)

#define __SYCL_PROCESS_BINARY_OPASSIGN_OPS(PROCESS_OP, DELIMITER) \
          PROCESS_OP(std::plus<void>)          \
DELIMITER PROCESS_OP(std::minus<void>)         \
DELIMITER PROCESS_OP(std::multiplies<void>)    \
DELIMITER PROCESS_OP(std::divides<void>)       \
DELIMITER PROCESS_OP(std::modulus<void>)       \
DELIMITER PROCESS_OP(std::bit_and<void>)       \
DELIMITER PROCESS_OP(std::bit_or<void>)        \
DELIMITER PROCESS_OP(std::bit_xor<void>)       \
DELIMITER PROCESS_OP(ShiftLeft)                \
DELIMITER PROCESS_OP(ShiftRight)

#define __SYCL_PROCESS_UNARY_OPS(PROCESS_OP, DELIMITER) \
          PROCESS_OP(std::negate<void>)          \
DELIMITER PROCESS_OP(std::logical_not<void>)     \
DELIMITER PROCESS_OP(std::bit_not<void>)         \
DELIMITER PROCESS_OP(UnaryPlus)
// clang-format on

// Need to separate binop/opassign because const vec swizzles don't have the
// latter.
template <typename Lhs, typename Rhs, typename Impl, typename DataT,
          typename Op, typename = void>
struct NonTemplateBinaryOpMixin {};
template <typename Lhs, typename Rhs, typename DataT, typename Op,
          typename = void>
struct NonTemplateBinaryOpAssignMixin {};

template <typename VecT, int... Indexes> class Swizzle;

template <typename Self, typename VecT, typename DataT, int N, typename Op,
          typename = void>
struct SwizzleTemplateBinaryOpMixin {};
template <typename Self, typename VecT, typename DataT, int N, typename Op,
          typename = void>
struct SwizzleTemplateBinaryOpAssignMixin {};

#define __SYCL_BINARY_OP_MIXIN(OP, BINOP)                                      \
  template <typename Lhs, typename Rhs, typename Impl, typename DataT>         \
  struct NonTemplateBinaryOpMixin<                                             \
      Lhs, Rhs, Impl, DataT, OP,                                               \
      std::enable_if_t<is_op_available<OP, DataT>>> {                          \
    friend auto operator BINOP(const Lhs &lhs, const Rhs &rhs) {               \
      return Impl{}(lhs, rhs, OP{});                                           \
    }                                                                          \
  };                                                                           \
  template <typename Self, typename VecT, typename DataT, int N>               \
  struct SwizzleTemplateBinaryOpMixin<                                         \
      Self, VecT, DataT, N, OP,                                                \
      std::enable_if_t<is_op_available<OP, DataT>>> {                          \
    template <typename OtherVecT, int... OtherIndexes,                         \
              typename = std::enable_if_t<                                     \
                  std::is_same_v<DataT, typename VecT::element_type> &&        \
                  N == sizeof...(OtherIndexes)>>                               \
    friend auto                                                                \
    operator BINOP(const Self &lhs,                                            \
                   const Swizzle<OtherVecT, OtherIndexes...> &rhs) {           \
      using ResultVec = vec<DataT, N>;                                         \
      return OP{}(static_cast<ResultVec>(lhs), static_cast<ResultVec>(rhs));   \
    }                                                                          \
    template <typename OtherVecT, int... OtherIndexes,                         \
              typename = std::enable_if_t<                                     \
                  std::is_same_v<DataT, typename VecT::element_type> &&        \
                  N == sizeof...(OtherIndexes) &&                              \
                  std::is_const_v<VecT> != std::is_const_v<OtherVecT>>>        \
    friend auto operator BINOP(const Swizzle<OtherVecT, OtherIndexes...> &lhs, \
                               const Self &rhs) {                              \
      using ResultVec = vec<DataT, N>;                                         \
      return OP{}(static_cast<ResultVec>(lhs), static_cast<ResultVec>(rhs));   \
    }                                                                          \
  };

#define __SYCL_BINARY_OP_AND_OPASSIGN_MIXIN(OP, BINOP, OPASSIGN)               \
  __SYCL_BINARY_OP_MIXIN(OP, BINOP)                                            \
  template <typename Lhs, typename Rhs, typename DataT>                        \
  struct NonTemplateBinaryOpAssignMixin<                                       \
      Lhs, Rhs, DataT, OP, std::enable_if_t<is_op_available<OP, DataT>>> {     \
    friend const Lhs &operator OPASSIGN(const Lhs & lhs, const Rhs & rhs) {    \
      lhs = OP{}(lhs, rhs);                                                    \
      return lhs;                                                              \
    }                                                                          \
  };                                                                           \
  template <typename Self, typename VecT, typename DataT, int N>               \
  struct SwizzleTemplateBinaryOpAssignMixin<                                   \
      Self, VecT, DataT, N, OP,                                                \
      std::enable_if_t<is_op_available<OP, DataT>>> {                          \
    template <typename OtherVecT, int... OtherIndexes,                         \
              typename = std::enable_if_t<                                     \
                  std::is_same_v<DataT, typename VecT::element_type> &&        \
                  N == sizeof...(OtherIndexes)>>                               \
    friend const Self &                                                        \
    operator OPASSIGN(const Self & lhs,                                        \
                      const Swizzle<OtherVecT, OtherIndexes...> &rhs) {        \
      using ResultVec = vec<DataT, N>;                                         \
      return OP{}(static_cast<ResultVec>(lhs), static_cast<ResultVec>(rhs));   \
    }                                                                          \
    template <typename OtherVecT, int... OtherIndexes,                         \
              typename = std::enable_if_t<                                     \
                  std::is_same_v<DataT, typename VecT::element_type> &&        \
                  N == sizeof...(OtherIndexes) &&                              \
                  std::is_const_v<VecT> != std::is_const_v<OtherVecT>>>        \
    friend auto                                                                \
    operator OPASSIGN(const Swizzle<OtherVecT, OtherIndexes...> &lhs,          \
                      const Self &rhs) {                                       \
      using ResultVec = vec<DataT, N>;                                         \
      return OP{}(static_cast<ResultVec>(lhs), static_cast<ResultVec>(rhs));   \
    }                                                                          \
  };

template <typename T, typename Impl, typename DataT, typename Op,
          typename = void>
struct UnaryOpMixin {};

#define __SYCL_UNARY_OP_MIXIN(OP, UOP)                                         \
  template <typename T, typename Impl, typename DataT>                         \
  struct UnaryOpMixin<T, Impl, DataT, OP,                                      \
                      std::enable_if_t<is_op_available<OP, DataT>>> {          \
    friend auto operator UOP(const T &x) { return Impl{}(x, OP{}); }           \
  };

// clang-format off
  __SYCL_BINARY_OP_AND_OPASSIGN_MIXIN(std::plus<void>       , +, +=)
  __SYCL_BINARY_OP_AND_OPASSIGN_MIXIN(std::minus<void>      , -, -=)
  __SYCL_BINARY_OP_AND_OPASSIGN_MIXIN(std::multiplies<void> , *, *=)
  __SYCL_BINARY_OP_AND_OPASSIGN_MIXIN(std::divides<void>    , /, /=)
  __SYCL_BINARY_OP_AND_OPASSIGN_MIXIN(std::modulus<void>    , %, %=)

  __SYCL_BINARY_OP_AND_OPASSIGN_MIXIN(std::bit_and<void>    , &, &=)
  __SYCL_BINARY_OP_AND_OPASSIGN_MIXIN(std::bit_or<void>     , |, |=)
  __SYCL_BINARY_OP_AND_OPASSIGN_MIXIN(std::bit_xor<void>    , ^, ^=)

  __SYCL_BINARY_OP_MIXIN(std::equal_to<void>                , ==)
  __SYCL_BINARY_OP_MIXIN(std::not_equal_to<void>            , !=)
  __SYCL_BINARY_OP_MIXIN(std::less<void>                    , <)
  __SYCL_BINARY_OP_MIXIN(std::greater<void>                 , >)
  __SYCL_BINARY_OP_MIXIN(std::less_equal<void>              , <=)
  __SYCL_BINARY_OP_MIXIN(std::greater_equal<void>           , >=)

  __SYCL_BINARY_OP_MIXIN(std::logical_and<void>             , &&)
  __SYCL_BINARY_OP_MIXIN(std::logical_or<void>              , ||)

  // TODO: versions for std::byte
  __SYCL_BINARY_OP_AND_OPASSIGN_MIXIN(ShiftLeft             , <<, <<=)
  __SYCL_BINARY_OP_AND_OPASSIGN_MIXIN(ShiftRight            , >>, >>=)

  __SYCL_UNARY_OP_MIXIN(std::negate<void>                   , -)
  __SYCL_UNARY_OP_MIXIN(std::logical_not<void>              , !)
  __SYCL_UNARY_OP_MIXIN(std::bit_not<void>                  , ~)
  __SYCL_UNARY_OP_MIXIN(UnaryPlus                           , +)
// clang-format on

#undef __SYCL_OP_MIXIN
#undef __SYCL_BINARY_OP_AND_OPASSIGN_MIXIN
#undef __SYCL_BINARY_OP_MIXIN

#define __SYCL_COMMA ,

// clang-format off
#define __SYCL_MIXIN_FOR_BINARY(OP)                                            \
public NonTemplateBinaryOpMixin<Lhs, Rhs, Impl, DataT, OP>

#define __SYCL_MIXIN_FOR_BINARY_OPASSIGN(OP)                                   \
public NonTemplateBinaryOpAssignMixin<Lhs, Rhs, DataT, OP>

#define __SYCL_MIXIN_FOR_TEMPLATE_BINARY(OP)                                   \
public SwizzleTemplateBinaryOpMixin<Self, VecT, DataT, N, OP>

#define __SYCL_MIXIN_FOR_TEMPLATE_BINARY_OPASSIGN(OP)                          \
public SwizzleTemplateBinaryOpAssignMixin<Self, VecT, DataT, N, OP>

#define __SYCL_MIXIN_FOR_UNARY(OP)                                             \
public UnaryOpMixin<T, Impl, DataT, OP>
// clang-format on

template <typename Lhs, typename Rhs, typename Impl, typename DataT>
struct NonTemplateBinaryOpsMixin
    : __SYCL_PROCESS_BINARY_OPS(__SYCL_MIXIN_FOR_BINARY, __SYCL_COMMA) {};

template <typename Lhs, typename Rhs, typename DataT>
struct NonTemplateBinaryOpAssignOpsMixin
    : __SYCL_PROCESS_BINARY_OPASSIGN_OPS(__SYCL_MIXIN_FOR_BINARY_OPASSIGN,
                                         __SYCL_COMMA) {};

template <typename Self, typename VecT, typename DataT, int N>
struct SwizzleTemplateBinaryOpsMixin
    : __SYCL_PROCESS_BINARY_OPS(__SYCL_MIXIN_FOR_TEMPLATE_BINARY,
                                __SYCL_COMMA) {};

template <typename Self, typename VecT, typename DataT, int N>
struct SwizzleTemplateBinaryOpAssignOpsMixin
    : __SYCL_PROCESS_BINARY_OPASSIGN_OPS(
          __SYCL_MIXIN_FOR_TEMPLATE_BINARY_OPASSIGN, __SYCL_COMMA) {};

template <typename T, typename Impl, typename DataT>
struct UnaryOpsMixin
    : __SYCL_PROCESS_UNARY_OPS(__SYCL_MIXIN_FOR_UNARY, __SYCL_COMMA) {};

#undef __SYCL_MIXIN_FOR_UNARY
#undef __SYCL_MIXIN_FOR_TEMPLATE_BINARY_OPASSIGN
#undef __SYCL_MIXIN_FOR_BINARY_OPASSIGN
#undef __SYCL_MIXIN_FOR_TEMPLATE_BINARY
#undef __SYCL_MIXIN_FOR_BINARY

#undef __SYCL_COMMA
#undef __SYCL_PROCESS_BINARY_OPS
#undef __SYCL_PROCESS_UNARY_OPS

struct SwizzleImpl {
private:
  template <typename T> static constexpr int num_elements() {
    if constexpr (is_vec_or_swizzle_v<T>)
      return T::size();
    else
      return 1;
  }

public:
  template <typename T0, typename T1, typename OpTy>
  auto operator()(const T0 &Lhs, const T1 &Rhs, OpTy &&Op) {
    static_assert(std::is_same_v<get_elem_type_t<T0>, get_elem_type_t<T1>>);
    constexpr auto N = std::max(num_elements<T0>(), num_elements<T1>());
    using ResultVec = vec<get_elem_type_t<T0>, N>;
    return Op(static_cast<ResultVec>(Lhs), static_cast<ResultVec>(Rhs));
  }
  template <typename T, typename OpTy> auto operator()(const T &X, OpTy &&Op) {
    using ResultVec = vec<typename T::element_type, T::size()>;
    return Op(static_cast<ResultVec>(X));
  }
};

template <typename Self, typename VecT, int N>
struct SwizzleBaseMixins
    : public NonTemplateBinaryOpsMixin<Self, typename VecT::element_type,
                                       SwizzleImpl,
                                       typename VecT::element_type>,
      public NonTemplateBinaryOpsMixin<typename VecT::element_type, Self,
                                       SwizzleImpl,
                                       typename VecT::element_type>,
      public NonTemplateBinaryOpsMixin<
          Self, vec<typename VecT::element_type, N>, SwizzleImpl,
          typename VecT::element_type>,
      public NonTemplateBinaryOpsMixin<vec<typename VecT::element_type, N>,
                                       Self, SwizzleImpl,
                                       typename VecT::element_type>,
      public UnaryOpsMixin<Self, SwizzleImpl, typename VecT::element_type>,
      public SwizzleTemplateBinaryOpsMixin<Self, VecT,
                                           typename VecT::element_type, N> {};

template <typename Self, typename VecT, int N>
struct SwizzleOpAssignMixins
    : public SwizzleBaseMixins<Self, VecT, N>,
      public NonTemplateBinaryOpAssignOpsMixin<
          Self, typename VecT::element_type, typename VecT::element_type>,
      public NonTemplateBinaryOpAssignOpsMixin<
          Self, vec<typename VecT::element_type, N>,
          typename VecT::element_type>,
      public SwizzleTemplateBinaryOpAssignOpsMixin<
          Self, VecT, typename VecT::element_type, N> {};

template <typename Self, typename DataT, typename = void>
struct PrefixIncDecMixin {};
template <typename Self, typename DataT>
struct PrefixIncDecMixin<Self, DataT,
                         std::enable_if_t<!std::is_same_v<bool, DataT>>> {
  friend const Self &operator++(const Self &x) {
    x += DataT{1};
    return x;
  }
  friend const Self &operator--(const Self &x) {
    x -= DataT{1};
    return x;
  }
};

template <typename Self, typename DataT, typename = void>
struct PrefixPostfixIncDecMixin {};
template <typename Self, typename DataT>
struct PrefixPostfixIncDecMixin<Self, DataT,
                                std::enable_if_t<!std::is_same_v<bool, DataT>>>
    : public PrefixIncDecMixin<Self, DataT> {
  friend auto operator++(const Self &x, int) {
    auto tmp = +x;
    x += DataT{1};
    return tmp;
  }
  friend auto operator--(const Self &x, int) {
    auto tmp = +x;
    x -= DataT{1};
    return tmp;
  }
};
} // namespace detail
} // namespace _V1
} // namespace sycl
