#include <sycl/sycl.hpp>
using namespace sycl;

template <int Dims> auto get_global_range(range<Dims> Range) { return Range; }
template <int Dims> auto get_global_range(nd_range<Dims> NDRange) {
  return NDRange.get_global_range();
}

template <int Dims, bool WithOffset>
auto get_global_id(item<Dims, WithOffset> Item) {
  return Item.get_id();
}
template <int Dims> auto get_global_id(nd_item<Dims> NDItem) {
  return NDItem.get_global_id();
}

template <int Dims> auto get_global_id(id<Dims> Id) { return Id; }

// We can select strategy explicitly so no need to test all combinations of
// types/operations.
using T = int;
using BinOpTy = std::plus<T>;

// On Windows, allocating new memory and then initializing it is slow for some
// reason (not related to reductions). Try to re-use the same memory between
// test cases.
struct RedStorage {
  RedStorage(queue &q) : q(q), Ptr(malloc_device<T>(1, q)), Buf(1) {}
  ~RedStorage() { free(Ptr, q); }

  template <bool UseUSM> auto get() {
    if constexpr (UseUSM)
      return Ptr;
    else
      return Buf;
  }
  queue &q;
  T *Ptr;
  buffer<T, 1> Buf;
};

template <bool UseUSM, bool InitToIdentity,
          detail::reduction::strategy Strategy, typename RangeTy>
struct Name {};

template <bool UseUSM, bool InitToIdentity,
          detail::reduction::strategy Strategy, typename RangeTy>
static void test(RedStorage &Storage, RangeTy Range) {
  queue &q = Storage.q;

  T Init{19};

  auto Red = Storage.get<UseUSM>();
  auto GetRedAcc = [&](handler &cgh) {
    if constexpr (UseUSM)
      return Red;
    else
      return accessor{Red, cgh};
  };

  q.submit([&](handler &cgh) {
     auto RedAcc = GetRedAcc(cgh);
     cgh.single_task([=]() { RedAcc[0] = Init; });
   }).wait();

  q.submit([&](handler &cgh) {
     auto RedSycl = [&]() {
       if constexpr (UseUSM)
         if constexpr (InitToIdentity)
           return reduction(Red, BinOpTy{},
                            property::reduction::initialize_to_identity{});
         else
           return reduction(Red, BinOpTy{});
       else if constexpr (InitToIdentity)
         return reduction(Red, cgh, BinOpTy{},
                          property::reduction::initialize_to_identity{});
       else
         return reduction(Red, cgh, BinOpTy{});
     }();
     detail::reduction_parallel_for<
         Name<UseUSM, InitToIdentity, Strategy, RangeTy>, Strategy>(
         cgh, Range, ext::oneapi::experimental::detail::empty_properties_t{},
         RedSycl, [=](auto Item, auto &Red) { Red.combine(T{1}); });
   }).wait();

  auto *Result = malloc_shared<T>(1, q);
  q.submit([&](handler &cgh) {
     auto RedAcc = GetRedAcc(cgh);
     cgh.single_task([=]() { *Result = RedAcc[0]; });
   }).wait();

  auto N = get_global_range(Range).size();
  int Expected = InitToIdentity ? N : Init + N;
#if defined(__PRETTY_FUNCTION__)
  std::cout << __PRETTY_FUNCTION__;
#elif defined(__FUNCSIG__)
  std::cout << __FUNCSIG__;
#endif
  std::cout << ": " << *Result << ", expected " << Expected << std::endl;
  assert(*Result == Expected);

  free(Result, q);
}

template <bool UseUSM, bool InitToIdentity, typename RangeTy>
void testAllStrategies(RedStorage &Storage, RangeTy Range) {
  detail::loop<(int)detail::reduction::strategy::multi>([&](auto Id) {
    // Skip auto_select == 0.
    constexpr auto Strategy = detail::reduction::strategy{Id + 1};
    test<UseUSM, InitToIdentity, Strategy>(Storage, Range);
  });
}

template <typename RangeTy>
void testRange(RedStorage &Storage, RangeTy Range) {
  testAllStrategies<true, true>(Storage, Range);
  testAllStrategies<true, false>(Storage, Range);
  testAllStrategies<false, true>(Storage, Range);
  testAllStrategies<false, false>(Storage, Range);
}
