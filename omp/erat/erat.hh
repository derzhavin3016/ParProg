#ifndef __OMP_ERAT_ERAT_HH__
#define __OMP_ERAT_ERAT_HH__

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <numbers>
#include <numeric>
#include <omp.h>
#include <string_view>
#include <vector>

#include "timer.hh"

using Numbers = std::vector<std::uint64_t>;
using EratFunc = std::function<Numbers(std::uint64_t)>;


inline Numbers seqErat(std::uint64_t num)
{
  assert(num >= 2);

  Numbers res;
  std::uint64_t log2n_rounded = std::bit_width(num) - 1;
  res.reserve(num * std::numbers::log2e_v<double> / log2n_rounded);

  std::vector<bool> sieve;
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

inline Numbers seqOddErat(std::uint64_t num)
{
  assert(num >= 2);

  Numbers res;
  std::uint64_t log2n_rounded = std::bit_width(num) - 1;
  res.reserve(num * std::numbers::log2e_v<double> / log2n_rounded);
  res.push_back(2);

  std::vector<bool> sieve;
  sieve.resize(((num - 1) >> 1) + 1, true);

  for (std::uint64_t cand = 3, cand_sq = cand * cand; cand_sq <= num;
       cand += 2, cand_sq = cand * cand)
  {
    if (!sieve[cand >> 1])
      continue;

    for (; cand_sq <= num; cand_sq += 2 * cand)
      sieve[cand_sq >> 1] = false;
  }

  for (std::uint64_t i = 1; i < sieve.size(); ++i)
    if (sieve[i])
      res.push_back((i << 1) + 1);

  return res;
}

#endif /* __OMP_ERAT_ERAT_HH__ */
