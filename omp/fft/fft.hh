#ifndef __OMP_FFT_FFT_HH__
#define __OMP_FFT_FFT_HH__

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <concepts>
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

constexpr Dbl thres = 1e-6;

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

inline Dbl runFFT(const Vec &inp, Vec &res, FFTfunc func)
{
  timer::Timer tim;
  res = func(inp);
  return tim.elapsed_mcs() / 1000.0l;
}

inline std::pair<Dbl, bool> testFFT(const Vec &inp, const Vec &ans,
                                    FFTfunc func)
{
  Vec res;
  auto elapsed = runFFT(inp, res, func);

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
  assert((N & (N - 1)) == 0);
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

inline Vec ctParFFT(const Vec &inp)
{
  auto N = inp.size();
  assert((N & (N - 1)) == 0);
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
  Vec res_even, res_odd;

#pragma omp parallel
  {
#pragma omp single nowait
    {
#pragma omp task
      res_even = ctFFT(even);
#pragma omp task
      res_odd = ctFFT(odd);
    }
  }
  Vec res(N);

#pragma omp parallel for
  for (std::size_t i = 0; i < hN; ++i)
  {
    auto wiN = calcRot(i, N);
    res[i] = res_even[i] + wiN * res_odd[i];
    res[i + hN] = res_even[i] - wiN * res_odd[i];
  }

  return res;
}

inline std::size_t bitRev(std::size_t val, std::size_t N)
{
  assert((N & (N - 1)) == 0);
  auto log2N = std::bit_width(N) - 1;

  std::size_t res = 0;
  for (std::size_t i = 0; i < log2N; ++i)
  {
    res <<= 1;
    res |= (val & 0x1);
    val >>= 1;
  }

  return res;
}

inline Vec ctFFTOpt(const Vec &inp)
{
  auto N = inp.size();
  auto log2N = std::bit_width(N) - 1;
  Vec res(N);
  Vec valW(N);

  for (std::size_t k = 0; k < N; ++k)
    valW[k] = calcRot(k, N);

  for (std::size_t i = 0; i < N; ++i)
    res[bitRev(i, N)] = inp[i];

  for (std::size_t stage = 1; stage <= log2N; ++stage)
  {
    std::size_t base = 1 << stage, baseH = base >> 1;

    for (std::size_t start = 0; start < N; start += base)
    {
      for (std::size_t j = 0; j <= baseH - 1; ++j)
      {
        auto w = valW[(j << (log2N - stage)) % N];

        auto &bef = res[start + j];
        auto &aft = res[start + j + baseH];

        auto noRot = bef;
        auto rot = w * aft;

        bef = noRot + rot;
        aft = noRot - rot;
      }
    }
  }

  return res;
}

inline Vec ctFFTParOpt(const Vec &inp)
{
  auto N = inp.size();
  auto log2N = std::bit_width(N) - 1;
  Vec res(N);
  Vec valW(N);

#pragma omp parallel for
  for (std::size_t k = 0; k < N; ++k)
    valW[k] = calcRot(k, N);

#pragma omp parallel for
  for (std::size_t i = 0; i < N; ++i)
    res[bitRev(i, N)] = inp[i];

  for (std::size_t stage = 1; stage <= log2N; ++stage)
  {
    std::size_t base = 1 << stage, baseH = base >> 1;

#pragma omp parallel for
    for (std::size_t start = 0; start < N; start += base)
    {
#pragma omp parallel for
      for (std::size_t j = 0; j <= baseH - 1; ++j)
      {
        auto w = valW[(j << (log2N - stage)) % N];

        auto &bef = res[start + j];
        auto &aft = res[start + j + baseH];

        auto noRot = bef;
        auto rot = w * aft;

        bef = noRot + rot;
        aft = noRot - rot;
      }
    }
  }

  return res;
}

Vec genData(
  std::size_t N, std::function<Compl(Compl)> func = [](Compl val) {
    return std::exp(Compl(0, 1) * val);
  })
{
  std::vector<std::size_t> ks(N);
  std::iota(ks.begin(), ks.end(), 0);
  Vec res(N);
  std::transform(ks.begin(), ks.end(), res.begin(), [func, N](auto k) {
    Compl arg = -2.0 * M_PI * k / N;
    return func(arg);
  });

  return res;
}

} // namespace fft

#endif /* __OMP_FFT_FFT_HH__ */
