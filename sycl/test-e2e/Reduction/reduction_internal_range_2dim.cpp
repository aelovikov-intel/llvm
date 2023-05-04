// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include "reduction_internal.hpp"

int main() {
  queue q;
  RedStorage Storage(q);
  testRange(Storage, range<2>{8, 8});
  return 0;
}
