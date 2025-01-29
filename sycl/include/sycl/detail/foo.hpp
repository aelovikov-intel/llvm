#pragma once
#include <sycl/backend_types.hpp>
#include <ostream>
#include <optional>

namespace sycl {
inline namespace _V1 {
inline std::ostream &operator<<(std::ostream &Out, backend be) {
  switch (be) {
  case backend::host:
    Out << "host";
    break;
  case backend::opencl:
    Out << "opencl";
    break;
  case backend::ext_oneapi_level_zero:
    Out << "ext_oneapi_level_zero";
    break;
  case backend::ext_oneapi_cuda:
    Out << "ext_oneapi_cuda";
    break;
  case backend::ext_oneapi_hip:
    Out << "ext_oneapi_hip";
    break;
  case backend::ext_oneapi_native_cpu:
    Out << "ext_oneapi_native_cpu";
    break;
  case backend::all:
    Out << "all";
  }
  return Out;
}
namespace detail {
template <typename T>
std::ostream &operator<<(std::ostream &os, std::optional<T> const &opt) {
  return opt ? os << opt.value() : os << "not set ";
}

} // namespace detail
} // namespace _V1
} // namespace sycl
