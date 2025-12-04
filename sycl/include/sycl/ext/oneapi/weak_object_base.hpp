//==------- weak_object_base.hpp --- SYCL weak objects base class ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/impl_utils.hpp> // for getSyclObjImpl
#include <sycl/exception.hpp>

#include <optional>
#include <utility>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::detail {
using namespace sycl::detail;
template <typename SYCLObjT, typename = void> class weak_object_base;

// Helper function for getting the underlying weak_ptr from a weak_object.
template <typename SYCLObjT>
decltype(weak_object_base<SYCLObjT>::MObjWeakPtr)
getSyclWeakObjImpl(const weak_object_base<SYCLObjT> &WeakObj) {
  return WeakObj.MObjWeakPtr;
}

// Common base class for weak_object.
template <typename SYCLObjT, typename> class weak_object_base {
public:
  using object_type = SYCLObjT;

  constexpr weak_object_base() noexcept : MObjWeakPtr() {}
  weak_object_base(const SYCLObjT &SYCLObj) noexcept
      : MObjWeakPtr(GetWeakImpl(SYCLObj)) {}
  weak_object_base(const weak_object_base &Other) noexcept = default;
  weak_object_base(weak_object_base &&Other) noexcept = default;

  weak_object_base &operator=(const weak_object_base &Other) noexcept = default;
  weak_object_base &operator=(weak_object_base &&Other) noexcept = default;

  void reset() noexcept { MObjWeakPtr.reset(); }
  void swap(weak_object_base &Other) noexcept {
    MObjWeakPtr.swap(Other.MObjWeakPtr);
  }

  bool expired() const noexcept { return MObjWeakPtr.expired(); }

#ifndef __SYCL_DEVICE_ONLY__
  bool owner_before(const SYCLObjT &Other) const noexcept {
    return MObjWeakPtr.owner_before(GetWeakImpl(Other));
  }
  bool owner_before(const weak_object_base &Other) const noexcept {
    return MObjWeakPtr.owner_before(Other.MObjWeakPtr);
  }
#else
  // On device calls to these functions are disallowed, so declare them but
  // don't define them to avoid compilation failures.
  bool owner_before(const SYCLObjT &Other) const noexcept;
  bool owner_before(const weak_object_base &Other) const noexcept;
#endif // __SYCL_DEVICE_ONLY__

protected:
#ifndef __SYCL_DEVICE_ONLY__
  // Store a weak variant of the impl in the SYCLObjT.
  typename std::remove_reference<decltype(sycl::detail::getSyclObjImpl(
      std::declval<SYCLObjT>()))>::type::weak_type MObjWeakPtr;
  // relies on <type_traits> from impl_utils.h

  static decltype(MObjWeakPtr) GetWeakImpl(const SYCLObjT &SYCLObj) {
    return sycl::detail::getSyclObjImpl(SYCLObj);
  }
#else
  // On device we may not have an impl, so we pad with an unused void pointer.
  std::weak_ptr<void> MObjWeakPtr;
  static std::weak_ptr<void> GetWeakImpl(const SYCLObjT &) { return {}; }
#endif // __SYCL_DEVICE_ONLY__

  template <class Obj>
  friend decltype(weak_object_base<Obj>::MObjWeakPtr)
  detail::getSyclWeakObjImpl(const weak_object_base<Obj> &WeakObj);
};

template <typename SYCLObjT>
class weak_object_base<
    SYCLObjT,
    std::enable_if_t<std::is_pointer_v<std::remove_reference_t<
        decltype(sycl::detail::getSyclObjImpl(std::declval<SYCLObjT>()))>>>> {
  using Impl =
      std::decay_t<decltype(*getSyclObjImpl(std::declval<SYCLObjT>()))>;
  friend SYCLObjT;

  Impl *impl = nullptr;

public:
  using object_type = SYCLObjT;

  constexpr weak_object_base() noexcept = default;
  weak_object_base(const SYCLObjT &dev) noexcept
      : impl(detail::getSyclObjImpl(dev)) {}
  weak_object_base(const weak_object_base &Other) noexcept = default;
  weak_object_base(weak_object_base &&Other) noexcept = default;

  weak_object_base &operator=(const SYCLObjT &Other) noexcept {
    this->impl = detail::getSyclObjImpl(Other);
    return *this;
  }
  weak_object_base &operator=(const weak_object_base &Other) noexcept = default;
  weak_object_base &operator=(weak_object_base &&Other) noexcept = default;

  bool expired() const noexcept { return impl == nullptr; }

  void reset() noexcept { impl = nullptr; }

#ifndef __SYCL_DEVICE_ONLY__
  std::optional<SYCLObjT> try_lock() const noexcept {
    if (!impl)
      return std::nullopt;
    return sycl::detail::createSyclObjFromImpl<SYCLObjT>(*impl);
  }
  SYCLObjT lock() const {
    std::optional<SYCLObjT> OptionalObj = try_lock();
    if (!OptionalObj)
      throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                            "Referenced object has expired.");
    return *OptionalObj;
  }
  bool owner_before(const SYCLObjT &Other) const noexcept {
    return impl < detail::getSyclObjImpl(Other);
  }
  bool owner_before(const weak_object_base &Other) const noexcept {
    return impl < Other.impl;
  }
#else
  // On SYCLObjT calls to these functions are disallowed, so declare them but
  // don't define them to avoid compilation failures.
  std::optional<SYCLObjT> try_lock() const noexcept;
  SYCLObjT lock() const;
  bool owner_before(const SYCLObjT &Other) const noexcept;
  bool owner_before(const weak_object_base &Other) const noexcept;
#endif // __SYCL_DEVICE_ONLY__
};
} // namespace ext::oneapi::detail
} // namespace _V1
} // namespace sycl
