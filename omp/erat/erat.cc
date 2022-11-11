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

Numbers seqErat(std::uint64_t num)
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

Numbers seqOddErat(std::uint64_t num)
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

bool isPrime(std::uint64_t num)
{
  if (num == 1)
    return false;
  if (num == 2)
    return true;
  for (std::uint64_t i = 2; i <= num >> 1; ++i)
    if (num % i == 0)
      return false;

  return true;
}

bool isPrimes(const Numbers &numbers)
{
  return std::all_of(numbers.begin(), numbers.end(), isPrime);
}

std::pair<long double, Numbers> measure(std::uint64_t num, EratFunc func)
{
  timer::Timer tim;

  auto res = func(num);

  auto elapsed = tim.elapsed_mcs();

  return {elapsed / 1000.0l, res};
}

void checkPrint(std::uint64_t num, EratFunc func, std::string_view name)
{
  auto [ms, res] = measure(num, func);
  bool passed = isPrimes(res);

  if (num < 20)
  {
    std::for_each(res.begin(), res.end(),
                  [](auto n) { std::cout << n << " "; });
    std::cout << std::endl;
  }

  std::cout << name << std::endl;
  std::cout << ms << " ms" << std::endl;
  std::cout << "Status: "
            << (passed ? "\033[1;32mPassed\033[0m" : "\033[1;31mFailed\033[0m")
            << std::endl;
}

int main(int argc, char *argv[])
{
  if (argc != 2)
  {
    std::cout << "Usage: " << argv[0] << " N" << std::endl;
    return 1;
  }

  auto N = std::atoll(argv[1]);
  if (N <= 0)
  {
    std::cout << "You must provide positive number" << std::endl;
    return 1;
  }

#define CHECK_ERAT(num, func) checkPrint((num), (func), #func);

  CHECK_ERAT(N, seqErat);
  CHECK_ERAT(N, seqOddErat);
}
