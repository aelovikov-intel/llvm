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

template <class Obj,
          typename = std::enable_if_t<std::is_base_of_v<ObjBaseTag, Obj>>>
const auto &getSyclObjImpl(const Obj &SyclObj) {
  assert(SyclObj.impl && "every constructor should create an impl");
  return SyclObj.impl;
}

template <class Obj,
          typename = std::enable_if_t<std::is_base_of_v<ObjBaseTag, Obj>>>
constexpr auto getCreator(const Obj &SyclObj) {
  return SyclObj.creator;
}

template <typename Impl, typename SyclObject> class ObjBase;
template <typename Impl, typename SyclObject>
class ObjBase<std::shared_ptr<Impl>, SyclObject> : ObjBaseTag {
protected:
  // TODO: Comment about usage and https://godbolt.org/z/WroY7fsYo
  using ObjBaseT = ObjBase;
  std::shared_ptr<Impl> impl;
  explicit ObjBase(std::shared_ptr<Impl> impl) : impl(std::move(impl)) {}

  template <class Obj, typename>
  friend const auto &getSyclObjImpl(const Obj &SyclObj);

  template <class Obj, typename>
  friend constexpr auto getCreator(const Obj &SyclObj);

  template <typename To>
  static To createObj(const std::shared_ptr<Impl> &impl) {
    return To{impl};
  }

  template <typename To> static To createObj(std::shared_ptr<Impl> &&impl) {
    return To{std::move(impl)};
  }

  template <typename To, typename Impl_ = Impl,
            typename =
                std::void_t<decltype(std::declval<Impl_>().shared_from_this())>>
  static To createObj(Impl &impl) {
    return To{impl.shared_from_this()};
  }

  template <typename To, typename From>
  friend To createSyclObjFromImpl(From &&from);

  struct Creator {
    template <typename To, typename From>
    static To create(From &&from) {
      static_assert(std::is_base_of_v<SyclObject, To>);
      return ObjBase::createObj<To>(std::forward<From>(from));
    }
  };

  static constexpr Creator creator{};
};

template <typename SyclObject, typename From>
SyclObject createSyclObjFromImpl(From &&from) {
  using Creator = decltype(getCreator(std::declval<SyclObject>()));
  return Creator::template create<SyclObject>(std::forward<From>(from));
}

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
