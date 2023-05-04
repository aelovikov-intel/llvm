// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include "reduction_internal.hpp"

int main() {
  queue q;
  RedStorage Storage(q);
  testRange(Storage, nd_range<1>{range<1>{7}, range<1>{7}});
  testRange(Storage, nd_range<1>{range<1>{3 * 3}, range<1>{3}});

  // TODO: Strategies historically adopted from sycl::range implementation only
  // support 1-Dim case.
  //
  // TestRange(nd_range<2>{range<2>{7, 3}, range<2> {7, 3}});
  // TestRange(nd_range<2>{range<2>{14, 9}, range<2> {7, 3}});
  return 0;
}
