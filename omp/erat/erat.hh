#ifndef __OMP_ERAT_ERAT_HH__
#define __OMP_ERAT_ERAT_HH__

#include <algorithm>
#include <cassert>
#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include <numbers>
#include <numeric>
#include <omp.h>
#include <sstream>
#include <string_view>
#include <vector>

#include "timer.hh"

using Numbers = std::vector<std::uint64_t>;
using Sieve = std::vector<std::uint8_t>;
using SieveIt = Sieve::iterator;
using EratNaiveFunc = std::function<Numbers(std::uint64_t)>;
using EratFunc = std::function<Sieve(std::uint64_t)>;

inline Numbers makeNumbers(std::uint64_t num)
{
  assert(num >= 2);
  Numbers res;
  std::uint64_t log2n_rounded = std::bit_width(num) - 1;
  res.reserve(num * std::numbers::log2e_v<double> / log2n_rounded);

  return res;
}

inline Numbers fromSieve(std::uint64_t num, const Sieve &sieve)
{
  auto nums = makeNumbers(num);
  nums.push_back(2);
  for (std::uint64_t i = 1; i < sieve.size(); ++i)
    if (sieve[i])
      nums.push_back((i << 1) | 1);

  return nums;
}

inline Numbers seqErat(std::uint64_t num)
{
  assert(num >= 2);

  Numbers res;
  std::uint64_t log2n_rounded = std::bit_width(num) - 1;
  res.reserve(num * std::numbers::log2e_v<double> / log2n_rounded);

  Sieve sieve;
  sieve.resize(num + 1, true);
  sieve[0] = sieve[1] = false;

  std::uint64_t root_num = std::sqrt(num);
  for (std::uint64_t cand = 2; cand <= root_num; ++cand)
  {
    if (!sieve[cand])
      continue;

    res.push_back(cand);

    for (auto cand_sq = cand * cand; cand_sq <= num; cand_sq += cand)
      sieve[cand_sq] = false;
  }

  for (std::uint64_t i = root_num + 1; i <= num; ++i)
    if (sieve[i])
      res.push_back(i);

  return res;
}

inline Sieve seqOddErat(std::uint64_t num)
{
  assert(num >= 2);
  Sieve sieve;
  sieve.resize(((num - 1) >> 1) + 1, true);

  for (std::uint64_t cand = 3, cand_sq = cand * cand; cand_sq <= num;
       cand += 2, cand_sq = cand * cand)
  {
    if (!sieve[cand >> 1])
      continue;

    for (; cand_sq <= num; cand_sq += 2 * cand)
      sieve[cand_sq >> 1] = false;
  }

  return sieve;
}
inline void fillOddSieveRange(std::uint64_t from, std::uint64_t to,
                              const SieveIt begSieve)
{
  assert(from >= 3 && (from & 1));

  for (std::uint64_t i = 3; i * i <= to; i += 2)
  {
    auto minCand = ((from + i - 1) / i) * i;
    minCand = std::max(minCand, i * i);

    if ((minCand & 1) == 0)
      minCand += i;

    // #pragma omp parallel for
    for (std::uint64_t notPrime = minCand; notPrime < to; notPrime += i << 1)
      begSieve[notPrime >> 1] = false;
  }
}

constexpr std::uint64_t SLICE_SZ = 4096; // 1024;
static_assert((SLICE_SZ & 1) == 0);

inline Sieve parOddErat(std::uint64_t num)
{
  if (num < 2)
    return {};

  Sieve sieve((num >> 1) + (num & 1), true);

#pragma omp parallel for schedule(dynamic)
  for (std::uint64_t from = 3; from <= num; from += SLICE_SZ)
  {
    auto to = from + SLICE_SZ;
    to = std::min(to, num + 1);

    fillOddSieveRange(from, to, sieve.begin());
  }
  return sieve;
}

#endif /* __OMP_ERAT_ERAT_HH__ */
