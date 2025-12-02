//===- impl_utils.hpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cassert>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

namespace sycl {
inline namespace _V1 {
class handler;
namespace detail {
// To enable SFINAE to limit things to SYCL objects only.
struct ObjBaseTag {};

struct ImplUtils {
  template <class Obj>
  static const decltype(Obj::impl) &getSyclObjImpl(const Obj &SyclObj) {
    assert(SyclObj.impl && "every constructor should create an impl");
    return SyclObj.impl;
  }

  template <typename SyclObject, typename From>
  static SyclObject createSyclObjFromImpl(From &&from) {
    if constexpr (std::is_same_v<decltype(SyclObject::impl),
                                 std::shared_ptr<std::decay_t<From>>>)
      return SyclObject{from.shared_from_this()};
    else
      return SyclObject{std::forward<From>(from)};
  }
};

template <class Obj,
          typename = std::enable_if_t<std::is_base_of_v<ObjBaseTag, Obj>>>
const auto &getSyclObjImpl(const Obj &SyclObj) {
  return ImplUtils::getSyclObjImpl(SyclObj);
}

template <typename SyclObject, typename From>
SyclObject createSyclObjFromImpl(From &&from) {
  return ImplUtils::createSyclObjFromImpl<SyclObject>(std::forward<From>(from));
}


template <typename Impl, typename SyclObject> class ObjBase;
template <typename Impl, typename SyclObject>
class ObjBase<std::shared_ptr<Impl>, SyclObject> : ObjBaseTag {
  friend ImplUtils;

protected:
  // TODO: Comment about usage and https://godbolt.org/z/WroY7fsYo
  using ObjBaseT = ObjBase;
  std::shared_ptr<Impl> impl;
  explicit ObjBase(std::shared_ptr<Impl> impl) : impl(std::move(impl)) {}
};

template <typename T, bool SupportedOnDevice = true> struct sycl_obj_hash {
  size_t operator()(const T &Obj) const {
    if constexpr (SupportedOnDevice) {
      auto &Impl = sycl::detail::getSyclObjImpl(Obj);
      return std::hash<std::decay_t<decltype(Impl)>>{}(Impl);
    } else {
#ifdef __SYCL_DEVICE_ONLY__
      (void)Obj;
      return 0;
#else
      auto &Impl = sycl::detail::getSyclObjImpl(Obj);
      return std::hash<std::decay_t<decltype(Impl)>>{}(Impl);
#endif
    }
  }
};

} // namespace detail
} // namespace _V1
} // namespace sycl
