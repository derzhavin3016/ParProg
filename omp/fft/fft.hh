#ifndef __OMP_FFT_FFT_HH__
#define __OMP_FFT_FFT_HH__

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <omp.h>
#include <string_view>
#include <vector>

#include "timer.hh"

namespace fft
{

using Dbl = long double;
using Compl = std::complex<Dbl>;
using Vec = std::vector<Compl>;

using FFTfunc = std::function<Vec(const Vec &)>;

constexpr Dbl thres = std::numeric_limits<Dbl>::epsilon() * 1e7;

bool isZero(Compl a)
{
  return std::abs(a) < thres;
}

bool isEq(Compl a, Compl b)
{
  return isZero(a - b);
}

void outputVec(const Vec &vec, std::ostream &ost = std::cout);
void inputVec(Vec &vec, std::istream &ist = std::cin);

inline std::pair<Dbl, bool> testFFT(const Vec &inp, const Vec &ans,
                                    FFTfunc func)
{
  timer::Timer tim;
  auto res = func(inp);
  auto elapsed = tim.elapsed_ms();

  bool passed =
    std::equal(res.begin(), res.end(), ans.begin(), ans.end(), isEq);

  return {elapsed, passed};
}

inline void testPrintFFT(const Vec &inp, const Vec &ans, FFTfunc func,
                         std::string_view name)
{
  auto [ms, passed] = testFFT(inp, ans, func);

  std::cout << name << std::endl;
  std::cout << ms << " ms" << std::endl;
  std::cout << "Status: "
            << (passed ? "\033[1;32mPassed\033[0m" : "\033[1;31mFailed\033[0m")
            << std::endl;
}

inline Vec naiveFFT(const Vec &inp)
{
  Vec res(inp.size());

  Compl power = {0, -2.0 * M_PI / inp.size()};
  for (std::size_t k = 0; k < res.size(); ++k)
  {
    auto &Xk = res[k];
    for (std::size_t j = 0; j < res.size(); ++j)
      Xk += inp[j] * std::exp(power * static_cast<Dbl>(k * j));
  }
  return res;
}

inline Compl calcRot(std::size_t k, std::size_t N)
{
  if (k % N == 0)
    return 1;
  Dbl arg = -2 * M_PI * k / N;
  return std::exp(arg * Compl(0, 1));
}

inline Vec ctFFT(const Vec &inp)
{
  auto N = inp.size();
  if (N == 2)
    return {inp[0] + inp[1], inp[0] - inp[1]};

  auto hN = N >> 1;
  Vec even(hN);
  Vec odd(hN);

  for (std::size_t i = 0; i < hN; ++i)
  {
    even[i] = inp[2 * i];
    odd[i] = inp[2 * i + 1];
  }

  auto res_even = ctFFT(even);
  auto res_odd = ctFFT(odd);

  Vec res(N);

  for (std::size_t i = 0; i < hN; ++i)
  {
    auto wiN = calcRot(i, N);
    res[i] = res_even[i] + wiN * res_odd[i];
    res[i + hN] = res_even[i] - wiN * res_odd[i];
  }

  return res;
}

} // namespace fft

#endif /* __OMP_FFT_FFT_HH__ */
